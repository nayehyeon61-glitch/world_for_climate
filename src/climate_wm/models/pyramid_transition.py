"""so(3) pyramid transition on the stacked (graph V-Net) latent (stage 4).

One Shell-GNN runs on the stacked node set of all levels (the stacked mesh has
every level's multimesh edges plus parent-child edges), proposes a tangent
velocity u_i and a spin s_i at every node, and the velocity is folded into
the so(3) pyramid by hierarchical Killing projections:

    omega^(0)  12 patches   from the coarsest state level's nodes
    omega^(1)  42 patches   from the coarsest state level's nodes
    omega^(2)  162 patches  from the level-2 nodes, regularised towards the
                            parent omega^(1)  (and so on up to detail_level)

Details are relative rotations to the parent, delta^(k) = log(exp(-omega^(k-1)_parent) exp(omega^(k))),
and every state level is moved by the scale chain blended onto its nodes,

    g_i = exp(omega^(0)) exp(delta^(1)) ... exp(delta^(k_i)),   k_i = min(level_i, detail_level),

with the shell chain's `compose`. The projection is GraphCast's ball-query
sum aggregation written in the Lie algebra: torques of the nodes inside a
patch are summed and divided by the patch inertia.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..geometry.group_actions import clamp_rotation, so3_exp
from .graphs import WorldGraphs
from .latent import Latent, LatentEmbedding
from .layers import ShellProcessor
from .lie_generator import LieConfig


class PyramidLieGenerator(nn.Module):

    def __init__(self, graphs: WorldGraphs, num_shells: int, history: int,
                 forcing_dim: int, cs: int, cv_latent: int, d: int, cv: int,
                 hidden_layers: int, cfg: LieConfig):
        super().__init__()
        if graphs.pyramid is None:
            raise ValueError("the pyramid transition needs mesh.pyramid_levels")
        if cfg.shell_coupling != "independent":
            raise ValueError("the pyramid transition supports shell_coupling=independent only")
        self.cfg = cfg
        self.embed = LatentEmbedding(history, cs, cv_latent, forcing_dim,
                                     graphs.latent_struct_dim, num_shells, d, cv,
                                     hidden_layers)
        self.processor = ShellProcessor(d, cv, graphs.latent_mesh.edge_dim,
                                        cfg.layers, num_shells, hidden_layers)
        self.velocity = nn.Linear(d, cv)
        self.spin = nn.Linear(d, 1)
        for lin in (self.velocity, self.spin):  # start at the identity action
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        # CFL-like cap per state level: a fraction of that level's edge length.
        self.register_buffer("max_angle", torch.tensor(
            [cfg.max_rotation_edges * e for e in graphs.pyramid.max_edge]), persistent=False)

    # ------------------------------------------------------------- algebra
    def forward(self, graphs: WorldGraphs, history: Sequence[Latent],
                node_forcing: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Returns omega [B, S, sum_k P_k, 3] (all pyramid parts concatenated:
        omega^(0), delta^(1), ...), omega_node [B, S, N_stacked, 3] and the
        details [B, S, sum_{k>=1} P_k, 3] (the scale-chain joints)."""
        return self.algebra(graphs, *self.features(graphs, history, node_forcing))

    def features(self, graphs: WorldGraphs, history: Sequence[Latent],
                 node_forcing: Tensor) -> Tuple[Tensor, Tensor]:
        h, v = self.embed(history, node_forcing, graphs.latent_struct)
        return self.processor(graphs.latent_mesh, h, v)

    def algebra(self, graphs: WorldGraphs, h: Tensor, v: Tensor):
        with torch.autocast(h.device.type, enabled=False):  # so(3) algebra in fp32
            return self._algebra(graphs, h.float(), v.float())

    def _algebra(self, graphs: WorldGraphs, h: Tensor, v: Tensor):
        pyr = graphs.pyramid
        b, s = h.shape[:2]
        u = (self.velocity(h)[..., None] * v).sum(-2)  # tangent velocity [B, S, N, 3]
        spin = self.spin(h)  # [B, S, N, 1]
        u_levels = [x.flatten(0, 1) for x in pyr.split(u)]
        spin_levels = [x.flatten(0, 1) for x in pyr.split(spin)]
        omegas = []
        for k, src in zip(pyr.rot_levels, pyr.rot_source):
            source = pyr.pyramid[src]
            patches = source.patches[k]
            uf = u_levels[src]
            if k == 0:
                prior = source.root(uf)[:, None, :].expand(-1, patches.num_patches, -1)
            else:
                prior = omegas[k - 1][:, source.parent(k)]
            om = patches.killing_projection(uf, prior)
            om = om + patches.pool(spin_levels[src]) * patches.centres[None]
            omegas.append(om)
        parts = pyr.pyramid[-1].details(omegas)  # [omega^(0), delta^(1), ...]
        nodes = [pyr.pyramid[i].node_algebra(parts[:len(pyr.pyramid[i].levels)])
                 for i in range(pyr.num_levels)]
        omega_node = torch.cat(nodes, 1).unflatten(0, (b, s))
        parts = [p.unflatten(0, (b, s)) for p in parts]
        omega = torch.cat(parts, 2)
        joints = torch.cat(parts[1:], 2) if len(parts) > 1 else None
        return omega, omega_node, joints

    # ---------------------------------------------------------------- action
    def act(self, graphs: WorldGraphs, z: Latent, omega_node: Tensor) -> Latent:
        """Phi_{exp(dt omega)}(z), level by level with that level's cap."""
        pyr = graphs.pyramid
        b, s = z.h.shape[:2]
        hs, vs = [], []
        with torch.autocast(omega_node.device.type, enabled=False):
            for i, (h_i, v_i, w_i) in enumerate(zip(pyr.split(z.h.float()), pyr.split(z.v.float()),
                                                    pyr.split(omega_node.float()))):
                step = clamp_rotation(self.cfg.dt * w_i, float(self.max_angle[i]))
                rot = so3_exp(step).flatten(0, 1)
                h_, v_ = pyr.rotation[i](h_i.flatten(0, 1), v_i.flatten(0, 1), rot)
                hs.append(h_.unflatten(0, (b, s)))
                vs.append(v_.unflatten(0, (b, s)))
        return Latent(torch.cat(hs, 2), torch.cat(vs, 2))

    @staticmethod
    def split_parts(graphs: WorldGraphs, omega: Tensor) -> list:
        """omega [..., sum_k P_k, 3] -> [omega^(0), delta^(1), ...] per patch level."""
        return list(torch.split(omega, list(graphs.pyramid.patch_counts), dim=-2))
