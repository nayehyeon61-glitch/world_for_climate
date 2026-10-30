"""Shell-GNN encoder E_phi: physical fields X_t -> latent Z_t (down-scaling).

    grid fields  --split into shells-->  per-shell (scalars, wind vectors)
                 --grid2mesh-->          mesh (shared GNN + shell embedding)
                 --mesh processor-->     multimesh MP + vertical shell MP
                 --pool-->               coarser latent mesh
                 --head-->               Z_t = (h [B,S,N,Cs], V [B,S,N,Cv,3])

Wind components (u, v) at every level enter as tangent vectors in R^3, so the
encoder is SO(3)-equivariant up to the explicitly symmetry-breaking inputs
(static fields, latitude features when `break_symmetry=True`).
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor, nn

from ..data.graphcast_mapping import ChannelLayout
from ..geometry.tangent_frames import uv_to_xyz
from .graphs import WorldGraphs
from .latent import Latent
from .layers import MLP, BipartiteBlock, ShellProcessor, VectorLinear


@dataclasses.dataclass
class ShellGNNConfig:
    hidden_dim: int = 64  # d: scalar width inside the GNNs
    vector_dim: int = 16  # cv: vector channels inside the GNNs
    latent_scalars: int = 32  # Cs of Z
    latent_vectors: int = 8  # Cv of Z
    hidden_layers: int = 1  # hidden layers per MLP
    encoder_layers: int = 2  # message-passing steps on the mesh
    latent_layers: int = 1  # steps on the latent mesh (if pooled)
    decoder_layers: int = 2


class ShellGNNEncoder(nn.Module):

    def __init__(self, layout: ChannelLayout, graphs: WorldGraphs,
                 cfg: ShellGNNConfig):
        super().__init__()
        self.layout = layout
        d, cv, hl = cfg.hidden_dim, cfg.vector_dim, cfg.hidden_layers
        s = len(layout.shells)
        common = len(layout.forcings) + len(layout.static)
        self.in_scalar = nn.ModuleList([
            MLP(len(p.scalar) + len(q.scalar) + common + graphs.struct_dim, d, d, hl)
            for p, q in zip(layout.dynamic_shells, layout.diagnostic_shells)])
        self.in_vector = nn.ModuleList([
            VectorLinear(max(len(p.vector_u), 1), cv) for p in layout.dynamic_shells])
        self.shell_embed = nn.Parameter(0.02 * torch.randn(s, d))
        self.mesh_embed = MLP(graphs.struct_dim, d, d, hl)
        self.grid2mesh = BipartiteBlock(d, cv, graphs.grid2mesh.edge_dim, hl)
        self.processor = ShellProcessor(d, cv, graphs.mesh.edge_dim,
                                        cfg.encoder_layers, s, hl)
        if graphs.has_pooling:
            self.latent_embed = MLP(graphs.struct_dim, d, d, hl)
            self.pool = BipartiteBlock(d, cv, graphs.pool.edge_dim, hl)
            self.latent_processor = ShellProcessor(
                d, cv, graphs.latent_mesh.edge_dim, cfg.latent_layers, s, hl)
        self.to_scalars = nn.Linear(d, cfg.latent_scalars)
        self.to_vectors = VectorLinear(cv, cfg.latent_vectors)
        self.norm = nn.LayerNorm(cfg.latent_scalars, elementwise_affine=False)

    def _node_init(self, embed: nn.Module, struct: Tensor, b: int, cv: int):
        h = embed(struct)[None, None] + self.shell_embed[None, :, None]
        h = h.expand(b, -1, -1, -1)
        v = h.new_zeros(h.shape[:3] + (cv, 3))
        return h, v

    def grid_features(self, graphs: WorldGraphs, x: Tensor):
        """Per-shell grid node features: h [B, S, G, d], v [B, S, G, cv, 3]."""
        b = x.shape[0]
        flat = x.flatten(2).transpose(1, 2)  # [B, G, C]
        n_dyn, n_diag = len(self.layout.dynamic), len(self.layout.diagnostics)
        diag = flat[..., n_dyn:n_dyn + n_diag]
        common = flat[..., n_dyn + n_diag:]
        struct = graphs.grid_struct.expand(b, -1, -1)
        hs, vs = [], []
        for s, (part, dpart) in enumerate(zip(self.layout.dynamic_shells,
                                              self.layout.diagnostic_shells)):
            scal = torch.cat([flat[..., part.scalar], diag[..., dpart.scalar], common, struct], -1)
            hs.append(self.in_scalar[s](scal) + self.shell_embed[s])
            if part.vector_u:
                vec = uv_to_xyz(flat[..., part.vector_u], flat[..., part.vector_v],
                                graphs.grid_east, graphs.grid_north)
            else:
                vec = flat.new_zeros(b, flat.shape[1], 1, 3)
            vs.append(self.in_vector[s](vec))
        return torch.stack(hs, 1), torch.stack(vs, 1)

    def forward(self, graphs: WorldGraphs, x: Tensor) -> Latent:
        """x [B, C_enc, H, W] normalised, channels = [dynamic | diagnostics | forcings | static]."""
        b = x.shape[0]
        h_grid, v_grid = self.grid_features(graphs, x)
        cv = v_grid.shape[-2]

        h, v = self._node_init(self.mesh_embed, graphs.mesh_struct, b, cv)
        h, v = self.grid2mesh(graphs.grid2mesh, h_grid, v_grid, h, v)
        h, v = self.processor(graphs.mesh, h, v)
        if graphs.has_pooling:
            h0, v0 = self._node_init(self.latent_embed, graphs.latent_struct, b, cv)
            h, v = self.pool(graphs.pool, h, v, h0, v0)
            h, v = self.latent_processor(graphs.latent_mesh, h, v)
        z = Latent(self.norm(self.to_scalars(h)), self.to_vectors(v))
        return z.project(graphs.latent_pos)
