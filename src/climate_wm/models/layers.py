"""Geometric message passing on S^2 (PyTorch Geometric based).

Node state = (h, V): scalars h [B, N, d] and tangent vectors V [B, N, Cv, 3]
stored in ambient coordinates. One `GeometricInteraction` step follows the
GraphCast interaction network (edge update -> aggregate -> node update, all
residual, LayerNorm'd MLPs) and adds a vector channel that is moved between
tangent planes with the Levi-Civita transport T_{j->i}:

    e'_ij = e_ij + MLP([e_ij, h_i, h_j, <T V_j, d_ij>, |T V_j|, <V_i, d_ij>])
    M_ij  = a(e'_ij) * T V_j + b(e'_ij) * d_ij            (in T_{p_i} S^2)
    h'_i  = h_i + MLP([h_i, mean_j e'_ij])
    V'_i  = P_i(V_i + sigma(g(h'_i)) * W mean_j M_ij)

With `break_symmetry=False` the edge features are rotation invariant and the
layer is exactly SO(3) equivariant (checked in tests/test_layers.py); with
`break_symmetry=True` latitude / local-frame features are added because the
real atmosphere is not rotation invariant (Earth's axis, orography, sun).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing

from ..geometry import spherical_mesh, tangent_frames
from ..geometry.tangent_frames import project_tangent


class MLP(nn.Sequential):
    """GraphCast-style MLP: (Linear, SiLU) x n -> Linear -> LayerNorm."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int,
                 hidden_layers: int = 1, layer_norm: bool = True):
        layers = [nn.Linear(d_in, d_hidden), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.SiLU()]
        layers.append(nn.Linear(d_hidden, d_out))
        if layer_norm:
            layers.append(nn.LayerNorm(d_out))
        super().__init__(*layers)


class VectorLinear(nn.Module):
    """Channel mixing of vector features without bias (equivariant)."""

    def __init__(self, c_in: int, c_out: int, zero_init: bool = False):
        super().__init__()
        weight = torch.randn(c_out, c_in) / math.sqrt(max(c_in, 1))
        self.weight = nn.Parameter(torch.zeros_like(weight) if zero_init else weight)

    def forward(self, v: Tensor) -> Tensor:
        return torch.einsum("...ci,oc->...oi", v, self.weight)


class FourierFeatures(nn.Module):
    """Random Fourier embedding of the noise level (as in DIAMOND)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.register_buffer("weight", torch.randn(1, dim // 2))

    def forward(self, x: Tensor) -> Tensor:
        f = 2 * math.pi * x.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)


class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale/shift come from a per-sample condition vector
    (graph analogue of DIAMOND's AdaGroupNorm)."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift = self.linear(cond)[:, None, :].chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


def structural_features(xyz: np.ndarray, break_symmetry: bool) -> np.ndarray:
    """Static node features: [sin lat, cos lat, sin lon, cos lon] or [1]."""
    if not break_symmetry:
        return np.ones((len(xyz), 1), np.float32)
    lat, lon = spherical_mesh.xyz_to_lat_lon(xyz)
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)],
                    -1).astype(np.float32)


class GeoGraph(nn.Module):
    """Static (bipartite or homogeneous) graph on S^2 with geometric edge data.

    Buffers:
      edge_index [2, E]: row 0 senders (src), row 1 receivers (dst).
      transport [E, 3, 3]: T_{src->dst} parallel transport.
      disp [E, 3]: log_{p_dst}(p_src) / max length (tangent at the receiver).
      edge_attr [E, F]: raw edge features (normalised length, optional
        receiver-frame components, optional interpolation weight).
    """

    def __init__(self, src_xyz: np.ndarray, dst_xyz: np.ndarray,
                 senders: np.ndarray, receivers: np.ndarray,
                 break_symmetry: bool, edge_weight: Optional[np.ndarray] = None):
        super().__init__()
        p_j, p_i = src_xyz[senders], dst_xyz[receivers]
        disp = tangent_frames.log_map(p_i, p_j)
        length = np.linalg.norm(disp, axis=-1)
        scale = max(float(length.max()), 1e-6)
        feats = [length / scale]
        if break_symmetry:
            lat_i, lon_i = spherical_mesh.xyz_to_lat_lon(p_i)
            east, north = tangent_frames.east_north_frames(lat_i, lon_i)
            feats += [(disp * east).sum(-1) / scale, (disp * north).sum(-1) / scale]
        if edge_weight is not None:
            feats.append(edge_weight)
        self.num_src, self.num_dst = len(src_xyz), len(dst_xyz)
        self.register_buffer("edge_index", torch.tensor(
            np.stack([senders, receivers]), dtype=torch.long))
        self.register_buffer("transport", torch.tensor(
            tangent_frames.transport_matrices(p_j, p_i), dtype=torch.float32))
        self.register_buffer("disp", torch.tensor(disp / scale, dtype=torch.float32))
        self.register_buffer("edge_attr", torch.tensor(
            np.stack(feats, -1), dtype=torch.float32))
        self.register_buffer("pos_dst", torch.tensor(dst_xyz, dtype=torch.float32))

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def edge_dim(self) -> int:
        return self.edge_attr.shape[1]


class GeometricInteraction(MessagePassing):
    """One residual interaction-network step with transported vector features."""

    def __init__(self, d: int, cv: int, hidden_layers: int = 1,
                 cond_dim: Optional[int] = None, aggr: str = "mean"):
        super().__init__(aggr=aggr, node_dim=1)
        self.d, self.cv = d, cv
        self.edge_mlp = MLP(3 * d + 3 * cv, d, d, hidden_layers)
        self.edge_gate = nn.Linear(d, 2 * cv)
        self.node_mlp = MLP(2 * d, d, d, hidden_layers)
        self.vec_mix = VectorLinear(cv, cv)
        self.vec_gate = nn.Linear(d, cv)
        self.cond_norm = AdaLayerNorm(d, cond_dim) if cond_dim else None
        self.checkpoint = False  # recompute activations in backward (large grids)

    def forward(self, graph: GeoGraph, h_src: Tensor, v_src: Tensor,
                h_dst: Tensor, v_dst: Tensor, e: Tensor,
                cond: Optional[Tensor] = None):
        """h_* [B, N, d], v_* [B, N, cv, 3], e [B or 1, E, d] edge latents."""
        if self.checkpoint and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                self._forward, graph, h_src, v_src, h_dst, v_dst, e, cond,
                use_reentrant=False)
        return self._forward(graph, h_src, v_src, h_dst, v_dst, e, cond)

    def _forward(self, graph, h_src, v_src, h_dst, v_dst, e, cond):
        msg = self.edge_updater(graph.edge_index, h=(h_src, h_dst),
                                v=(v_src, v_dst), e=e, T=graph.transport,
                                disp=graph.disp)
        agg = self.propagate(graph.edge_index, msg=msg,
                             size=(graph.num_src, graph.num_dst))
        e_new = msg[..., :self.d]
        agg_e = agg[..., :self.d]
        agg_v = agg[..., self.d:].unflatten(-1, (self.cv, 3))
        h_in = self.cond_norm(h_dst, cond) if self.cond_norm is not None else h_dst
        h_new = h_dst + self.node_mlp(torch.cat([h_in, agg_e], -1))
        gate = torch.sigmoid(self.vec_gate(h_new))[..., None]
        v_new = v_dst + gate * self.vec_mix(agg_v)
        v_new = project_tangent(v_new, graph.pos_dst.expand(v_new.shape[0], -1, -1))
        return h_new, v_new, e_new

    def edge_update(self, h_i: Tensor, h_j: Tensor, v_i: Tensor, v_j: Tensor,
                    e: Tensor, T: Tensor, disp: Tensor) -> Tensor:
        tv_j = torch.einsum("eij,becj->beci", T, v_j)
        d = disp[None, :, None, :]
        invariants = torch.cat([(tv_j * d).sum(-1), tv_j.norm(dim=-1),
                                (v_i * d).sum(-1)], -1)
        e = e.expand(h_i.shape[0], -1, -1)
        e_new = e + self.edge_mlp(torch.cat([e, h_i, h_j, invariants], -1))
        a, b = self.edge_gate(e_new).chunk(2, dim=-1)
        m_v = a[..., None] * tv_j + b[..., None] * d
        return torch.cat([e_new, m_v.flatten(-2)], -1)

    def message(self, msg: Tensor) -> Tensor:
        return msg


class VerticalShellMixer(nn.Module):
    """Message passing along the vertical chain of shells at every node.

    Shells at the same node share the tangent plane, so vectors need no
    transport; neighbours are shell s-1 and s+1 (zero padded at the ends).
    """

    def __init__(self, d: int, cv: int, hidden_layers: int = 1):
        super().__init__()
        self.mlp = MLP(3 * d, d, d, hidden_layers)
        self.vec = nn.ModuleList([VectorLinear(cv, cv) for _ in range(3)])
        self.gate = nn.Linear(d, 3 * cv)

    @staticmethod
    def _neighbors(x: Tensor):
        pad = torch.zeros_like(x[:, :1])
        below = torch.cat([pad, x[:, :-1]], 1)
        above = torch.cat([x[:, 1:], pad], 1)
        return below, above

    def forward(self, h: Tensor, v: Tensor):
        """h [B, S, N, d], v [B, S, N, cv, 3]."""
        h_lo, h_hi = self._neighbors(h)
        h_new = h + self.mlp(torch.cat([h_lo, h, h_hi], -1))
        v_lo, v_hi = self._neighbors(v)
        gates = torch.sigmoid(self.gate(h_new)).chunk(3, dim=-1)
        v_new = v
        for g, lin, vv in zip(gates, self.vec, (v_lo, v, v_hi)):
            v_new = v_new + g[..., None] * lin(vv)
        return h_new, v_new


class ShellProcessor(nn.Module):
    """L x [shared GeometricInteraction on a mesh -> vertical shell mixing].

    All shells are processed by the same weights (shell identity comes from a
    shell embedding added upstream); shells are folded into the batch axis.
    """

    def __init__(self, d: int, cv: int, edge_dim: int, num_layers: int,
                 num_shells: int, hidden_layers: int = 1,
                 cond_dim: Optional[int] = None):
        super().__init__()
        self.edge_embed = MLP(edge_dim, d, d, hidden_layers)
        self.layers = nn.ModuleList([
            GeometricInteraction(d, cv, hidden_layers, cond_dim)
            for _ in range(num_layers)])
        self.vertical = nn.ModuleList([
            VerticalShellMixer(d, cv, hidden_layers) if num_shells > 1
            else nn.Identity() for _ in range(num_layers)])

    def forward(self, graph: GeoGraph, h: Tensor, v: Tensor,
                cond: Optional[Tensor] = None):
        b, s = h.shape[:2]
        if cond is not None:
            cond = cond.repeat_interleave(s, dim=0)
        e = self.edge_embed(graph.edge_attr)[None]
        hf, vf = h.flatten(0, 1), v.flatten(0, 1)
        for layer, vertical in zip(self.layers, self.vertical):
            hf, vf, e = layer(graph, hf, vf, hf, vf, e, cond)
            if not isinstance(vertical, nn.Identity):
                h_, v_ = vertical(hf.unflatten(0, (b, s)), vf.unflatten(0, (b, s)))
                hf, vf = h_.flatten(0, 1), v_.flatten(0, 1)
        return hf.unflatten(0, (b, s)), vf.unflatten(0, (b, s))


class BipartiteBlock(nn.Module):
    """Single message-passing hop between two node sets (grid2mesh, pooling,
    unpooling, mesh2grid). Shells are folded into the batch axis."""

    def __init__(self, d: int, cv: int, edge_dim: int, hidden_layers: int = 1):
        super().__init__()
        self.edge_embed = MLP(edge_dim, d, d, hidden_layers)
        self.layer = GeometricInteraction(d, cv, hidden_layers)

    def forward(self, graph: GeoGraph, h_src: Tensor, v_src: Tensor,
                h_dst: Tensor, v_dst: Tensor):
        """h_src [B, S, Ns, d] ... returns (h_dst', v_dst') [B, S, Nd, ...]."""
        b, s = h_src.shape[:2]
        e = self.edge_embed(graph.edge_attr)[None]
        h, v, _ = self.layer(graph, h_src.flatten(0, 1), v_src.flatten(0, 1),
                             h_dst.flatten(0, 1), v_dst.flatten(0, 1), e)
        return h.unflatten(0, (b, s)), v.unflatten(0, (b, s))
