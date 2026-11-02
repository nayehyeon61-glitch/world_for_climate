"""Graph V-Net encoder / decoder over the icosphere hierarchy (stage 4).

Encoder (contracting path, GraphCast conventions at every level):

    grid --grid2mesh--> level 3 --processor--> Z^(3)
             |            |  down (ball query, radius 0.6 x coarse edge)
             +--grid2level(2)--> level 2 --block--> Z^(2)
             |            |  down
             +--grid2level(1)--> level 1 --block--> Z^(1)

Every level receives the pooled finer level *and* a direct GraphCast-style
ball-query aggregation of the grid with the radius of its own edge length, so
a coarse state is an integration of the grid, not only a smoothed fine
latent. Decoder (expanding path):

    Z^(1) --block--> up (containing triangle) --+ skip Z^(2) --block--> up --+ skip Z^(3)
                                                                       --processor--> mesh2grid --> fields

Skips are the level states themselves (in a rollout: predicted states).
During training each sample's skips are dropped with probability
`skip_dropout`, so the decoder can also reconstruct from the coarsest level
alone (the P-skip arm keeps only that level as state).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.graphcast_mapping import ChannelLayout
from .graphs import WorldGraphs
from .latent import Latent
from .layers import MLP, BipartiteBlock, ShellProcessor, VectorLinear
from .shell_gnn_decoder import ShellGNNDecoder
from .shell_gnn_encoder import ShellGNNConfig, ShellGNNEncoder


class PyramidEncoder(ShellGNNEncoder):

    def __init__(self, layout: ChannelLayout, graphs: WorldGraphs,
                 cfg: ShellGNNConfig, level_layers: int = 1):
        super().__init__(layout, graphs, cfg)
        pyr = graphs.pyramid
        if pyr is None:
            raise ValueError("PyramidEncoder needs mesh.pyramid_levels")
        d, cv, hl = cfg.hidden_dim, cfg.vector_dim, cfg.hidden_layers
        s = len(layout.shells)
        coarse = range(len(pyr.path_levels) - 1)  # path levels below the mesh level
        self.level_embed = nn.ModuleList([MLP(pyr.struct(j).shape[1], d, d, hl) for j in coarse])
        self.down = nn.ModuleList([BipartiteBlock(d, cv, pyr.down[j].edge_dim, hl) for j in coarse])
        self.grid2level = nn.ModuleList([
            BipartiteBlock(d, cv, pyr.grid2level[j].edge_dim, hl) for j in coarse])
        self.block = nn.ModuleList([
            ShellProcessor(d, cv, pyr.mesh[j].edge_dim, level_layers, s, hl) for j in coarse])
        # Heads of the coarser state levels (the parent's head serves the finest).
        self.level_scalars = nn.ModuleList([nn.Linear(d, cfg.latent_scalars) for _ in pyr.levels[:-1]])
        self.level_vectors = nn.ModuleList([VectorLinear(cv, cfg.latent_vectors) for _ in pyr.levels[:-1]])

    def forward(self, graphs: WorldGraphs, x: Tensor) -> Latent:
        pyr = graphs.pyramid
        b = x.shape[0]
        h_grid, v_grid = self.grid_features(graphs, x)
        cv = v_grid.shape[-2]
        h, v = self._node_init(self.mesh_embed, graphs.mesh_struct, b, cv)
        h, v = self.grid2mesh(graphs.grid2mesh, h_grid, v_grid, h, v)
        h, v = self.processor(graphs.mesh, h, v)
        states = {pyr.path_levels[-1]: (h, v)}
        for j in reversed(range(len(pyr.path_levels) - 1)):
            h0, v0 = self._node_init(self.level_embed[j], pyr.struct(j), b, cv)
            h, v = self.down[j](pyr.down[j], h, v, h0, v0)  # pooled finer level
            h, v = self.grid2level[j](pyr.grid2level[j], h_grid, v_grid, h, v)  # grid ball query
            h, v = self.block[j](pyr.mesh[j], h, v)
            states[pyr.path_levels[j]] = (h, v)
        hs, vs = [], []
        for i, lv in enumerate(pyr.levels):
            h_i, v_i = states[lv]
            last = i == pyr.num_levels - 1
            head_s = self.to_scalars if last else self.level_scalars[i]
            head_v = self.to_vectors if last else self.level_vectors[i]
            z = Latent(self.norm(head_s(h_i)), head_v(v_i)).project(pyr.pos(pyr.path_index(lv)))
            hs.append(z.h)
            vs.append(z.v)
        return Latent(torch.cat(hs, 2), torch.cat(vs, 2))


class PyramidDecoder(ShellGNNDecoder):

    def __init__(self, layout: ChannelLayout, graphs: WorldGraphs,
                 cfg: ShellGNNConfig, level_layers: int = 1, skip_dropout: float = 0.5):
        super().__init__(layout, graphs, cfg)
        pyr = graphs.pyramid
        if pyr is None:
            raise ValueError("PyramidDecoder needs mesh.pyramid_levels")
        d, cv, hl = cfg.hidden_dim, cfg.vector_dim, cfg.hidden_layers
        s = len(layout.shells)
        self.skip_dropout = float(skip_dropout)
        n = len(pyr.path_levels)
        self.level_in_scalars = nn.ModuleList([
            MLP(cfg.latent_scalars + pyr.struct(pyr.path_index(lv)).shape[1], d, d, hl)
            for lv in pyr.levels])
        self.level_in_vectors = nn.ModuleList([
            VectorLinear(cfg.latent_vectors, cv) for _ in pyr.levels])
        self.level_embed = nn.ModuleList([MLP(pyr.struct(j).shape[1], d, d, hl) for j in range(1, n)])
        self.up = nn.ModuleList([BipartiteBlock(d, cv, pyr.up[j].edge_dim, hl) for j in range(n - 1)])
        # Coarse path levels get their own residual blocks; the finest level
        # uses the parent's `processor` (decoder_layers).
        self.block = nn.ModuleList([
            ShellProcessor(d, cv, pyr.mesh[j].edge_dim, level_layers, s, hl) for j in range(n - 1)])

    def _embed_level(self, pyr, i: int, zh, zv, b: int, s: int):
        j = pyr.path_index(pyr.levels[i])
        struct = pyr.struct(j).expand(b, s, -1, -1)
        h = self.level_in_scalars[i](torch.cat([zh[i], struct], -1)) + self.shell_embed[None, :, None]
        return h, self.level_in_vectors[i](zv[i])

    def forward(self, graphs: WorldGraphs, z: Latent) -> Tensor:
        pyr = graphs.pyramid
        b, s = z.h.shape[:2]
        zh, zv = pyr.split(z.h), pyr.split(z.v)
        mask = None
        if self.training and self.skip_dropout > 0 and pyr.num_levels > 1:
            keep = torch.rand(b, device=z.h.device) >= self.skip_dropout
            mask = keep.to(z.h.dtype)[:, None, None, None]
        h, v = self._embed_level(pyr, 0, zh, zv, b, s)
        h, v = self.block[0](pyr.mesh[0], h, v)
        for j in range(1, len(pyr.path_levels)):
            lv = pyr.path_levels[j]
            h0, v0 = self._node_init(self.level_embed[j - 1], pyr.struct(j), b)
            if lv in pyr.levels:  # skip from this level's state
                hs, vs = self._embed_level(pyr, pyr.levels.index(lv), zh, zv, b, s)
                if mask is not None:
                    hs, vs = hs * mask, vs * mask[..., None]
                h0, v0 = h0 + hs, v0 + vs
            h, v = self.up[j - 1](pyr.up[j - 1], h, v, h0, v0)
            proc = self.processor if j == len(pyr.path_levels) - 1 else self.block[j]
            h, v = proc(pyr.mesh[j], h, v)
        return self.readout(graphs, h, v)
