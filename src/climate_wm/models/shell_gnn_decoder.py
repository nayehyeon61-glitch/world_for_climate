"""Shell-GNN decoder D_psi: latent Z -> GraphCast target fields (up-scaling).

    Z (latent mesh) --latent MP--> --unpool--> mesh --mesh MP-->
    --mesh2grid--> per-shell heads -> scalars + (u, v) from tangent vectors

The output is the normalised target tensor [B, C_tgt, H, W] in the
`ChannelLayout.targets` order (GraphCast target variables on the lat-lon grid).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.graphcast_mapping import ChannelLayout
from ..geometry.tangent_frames import xyz_to_uv
from .graphs import WorldGraphs
from .latent import Latent
from .layers import MLP, BipartiteBlock, ShellProcessor, VectorLinear
from .shell_gnn_encoder import ShellGNNConfig


class ShellGNNDecoder(nn.Module):

    def __init__(self, layout: ChannelLayout, graphs: WorldGraphs,
                 cfg: ShellGNNConfig):
        super().__init__()
        self.layout = layout
        d, cv, hl = cfg.hidden_dim, cfg.vector_dim, cfg.hidden_layers
        s = len(layout.shells)
        self.cv = cv
        self.from_scalars = MLP(cfg.latent_scalars + graphs.struct_dim, d, d, hl)
        self.from_vectors = VectorLinear(cfg.latent_vectors, cv)
        self.shell_embed = nn.Parameter(0.02 * torch.randn(s, d))
        if graphs.has_pooling:
            self.latent_processor = ShellProcessor(
                d, cv, graphs.latent_mesh.edge_dim, cfg.latent_layers, s, hl)
            self.mesh_embed = MLP(graphs.struct_dim, d, d, hl)
            self.unpool = BipartiteBlock(d, cv, graphs.unpool.edge_dim, hl)
        self.processor = ShellProcessor(d, cv, graphs.mesh.edge_dim,
                                        cfg.decoder_layers, s, hl)
        self.grid_embed = MLP(graphs.struct_dim, d, d, hl)
        self.mesh2grid = BipartiteBlock(d, cv, graphs.mesh2grid.edge_dim, hl)
        self.out_scalar = nn.ModuleList([
            MLP(d, d, max(len(p.scalar), 1), hl, layer_norm=False)
            for p in layout.target_shells])
        self.out_vector = nn.ModuleList([
            VectorLinear(cv, max(len(p.vector_u), 1)) for p in layout.target_shells])
        order = []
        for p in layout.target_shells:
            order += p.scalar + p.vector_u + p.vector_v
        self.register_buffer("inverse_order", torch.argsort(torch.tensor(order)))

    def _node_init(self, embed: nn.Module, struct: Tensor, b: int):
        h = embed(struct)[None, None] + self.shell_embed[None, :, None]
        h = h.expand(b, -1, -1, -1)
        return h, h.new_zeros(h.shape[:3] + (self.cv, 3))

    def forward(self, graphs: WorldGraphs, z: Latent) -> Tensor:
        b, s, n = z.h.shape[:3]
        struct = graphs.latent_struct.expand(b, s, n, -1)
        h = self.from_scalars(torch.cat([z.h, struct], -1)) + self.shell_embed[None, :, None]
        v = self.from_vectors(z.v)
        if graphs.has_pooling:
            h, v = self.latent_processor(graphs.latent_mesh, h, v)
            h0, v0 = self._node_init(self.mesh_embed, graphs.mesh_struct, b)
            h, v = self.unpool(graphs.unpool, h, v, h0, v0)
        h, v = self.processor(graphs.mesh, h, v)
        return self.readout(graphs, h, v)

    def readout(self, graphs: WorldGraphs, h: Tensor, v: Tensor) -> Tensor:
        """Mesh features [B, S, N_mesh, ...] -> normalised target fields [B, C, H, W]."""
        b = h.shape[0]
        hg, vg = self._node_init(self.grid_embed, graphs.grid_struct, b)
        hg, vg = self.mesh2grid(graphs.mesh2grid, h, v, hg, vg)

        pieces = []
        for k, part in enumerate(self.layout.target_shells):
            if part.scalar:
                pieces.append(self.out_scalar[k](hg[:, k]))
            if part.vector_u:
                u, v_ = xyz_to_uv(self.out_vector[k](vg[:, k]), graphs.grid_east,
                                  graphs.grid_north)
                pieces += [u, v_]
        out = torch.cat(pieces, -1)[..., self.inverse_order]  # [B, G, C_tgt]
        hh, ww = graphs.grid_shape
        return out.transpose(1, 2).reshape(b, -1, hh, ww)
