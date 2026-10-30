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

Stage 5 (`model.composite.enabled`) widens the step to the composite group
(SO(3) x Vert) ⋉ Fib of `geometry.composite_group`: the same features also
give a vertical shift w per column (shells per step) and the physics-family
coefficients of the fibre generators, and the step is the Strang splitting
exp(dt/2 F) exp(dt w d/d eta) exp(dt omega) exp(dt/2 F). The shift head is
zero-initialised and every family has a gain that starts at zero (kept
non-negative for mixing and condensation by a straight-through projection),
so a composite model starts as the plain pyramid transition.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..geometry.composite_group import (FIB_FAMILIES, CompositeConfig, FibreCoefficients,
                                        FibreGenerators, split_step, vert_shift)
from ..geometry.group_actions import clamp_rotation, so3_exp
from .graphs import WorldGraphs
from .latent import Latent, LatentEmbedding
from .layers import ShellProcessor
from .lie_generator import LieConfig


@dataclasses.dataclass
class CompositeTerms:
    """Generators of one step besides the rotations (stage 5)."""
    shift: Optional[Tensor]  # [B, N] vertical shift in shells per step (None = Vert off)
    coef: FibreCoefficients  # physics-family coefficients

    def norm2(self) -> Tensor:
        """Mean squared generator size (the fibre and shift part of L_geo)."""
        parts = list(self.coef.norm2().values())
        if self.shift is not None:
            parts.append(self.shift.square().mean())
        return sum(parts) if parts else torch.zeros(())


def _nonneg(g: Tensor) -> Tensor:
    """max(g, 0) in the forward pass with the identity's gradient, so that a
    gain projected to zero can still come back."""
    return g + (g.clamp_min(0) - g).detach()


class PyramidLieGenerator(nn.Module):

    def __init__(self, graphs: WorldGraphs, num_shells: int, history: int,
                 forcing_dim: int, cs: int, cv_latent: int, d: int, cv: int,
                 hidden_layers: int, cfg: LieConfig,
                 composite: Optional[CompositeConfig] = None):
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
        self.composite = composite if composite is not None and composite.enabled else None
        if self.composite is not None:
            c = self.composite
            self.fibre = FibreGenerators(num_shells, cs, cv_latent, c)
            self.coef_head = nn.Linear(d, self.fibre.num_raw)
            self.gate_head = nn.Linear(d, 2) if c.diag_gates else None  # mixing, condensation
            self.fibre_gain = nn.Parameter(torch.zeros(len(FIB_FAMILIES)))
            self.shift = nn.Linear(d, 1) if c.vert else None
            if self.shift is not None:
                nn.init.zeros_(self.shift.weight)
                nn.init.zeros_(self.shift.bias)

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

    def generate(self, graphs: WorldGraphs, history: Sequence[Latent], node_forcing: Tensor
                 ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[CompositeTerms]]:
        """`forward` plus the stage 5 shift and fibre coefficients (None if off)."""
        h, v = self.features(graphs, history, node_forcing)
        omega, omega_node, joints = self.algebra(graphs, h, v)
        terms = self.composite_terms(h) if self.composite is not None else None
        return omega, omega_node, joints, terms

    def composite_terms(self, h: Tensor) -> CompositeTerms:
        """Processed features h [B, S, N, d] -> vertical shift and fibre coefficients."""
        with torch.autocast(h.device.type, enabled=False):
            h = h.float()
            gate_mix = gate_cond = None
            if self.gate_head is not None:  # learned gates on the features (which carry the
                g = torch.sigmoid(self.gate_head(h))  # encoded diagnostics in arm G2)
                gate_mix, gate_cond = g[..., 0].mean(1), g[..., 1]
            coef = self.fibre.coefficients(self.coef_head(h), gate_mix, gate_cond)
            gain = dict(zip(FIB_FAMILIES, self.fibre_gain))
            scale = lambda x, k: None if x is None else x * k
            coef = FibreCoefficients(
                radiation=scale(coef.radiation, gain["radiation"]),
                mixing=scale(coef.mixing, _nonneg(gain["mixing"])),
                surface=scale(coef.surface, gain["surface"]),
                condensation=scale(coef.condensation, _nonneg(gain["condensation"])))
            shift = self.shift(h)[..., 0].mean(1) if self.shift is not None else None
        return CompositeTerms(shift, coef)

    def algebra(self, graphs: WorldGraphs, h: Tensor, v: Tensor):
        with torch.autocast(h.device.type, enabled=False):  # so(3) algebra in fp32
            return self._algebra(graphs, h.float(), v.float())

    def _algebra(self, graphs: WorldGraphs, h: Tensor, v: Tensor):
        u = (self.velocity(h)[..., None] * v).sum(-2)  # tangent velocity [B, S, N, 3]
        return self.project(graphs, u, self.spin(h))

    def project(self, graphs: WorldGraphs, u: Tensor, spin: Optional[Tensor] = None):
        """Node velocities u [B, S, N, 3] (rad per step) and spins [B, S, N, 1]
        -> the pyramid (omega, omega_node, joints) as in `forward`. Also used for
        the generators implied by an observed wind field (stage 5 critic)."""
        pyr = graphs.pyramid
        b, s = u.shape[:2]
        if spin is None:
            spin = u.new_zeros(u.shape[:-1] + (1,))
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
        parts = [p.unflatten(0, (b, s)) for p in pyr.pyramid[-1].details(omegas)]
        omega_node = self.node_algebra(graphs, parts)
        omega = torch.cat(parts, 2)
        joints = torch.cat(parts[1:], 2) if len(parts) > 1 else None
        return omega, omega_node, joints

    @staticmethod
    def node_algebra(graphs: WorldGraphs, parts: Sequence[Tensor]) -> Tensor:
        """Scale chain [omega^(0), delta^(1), ...] (each [B, S, P_k, 3]) blended
        onto every state level's nodes: [B, S, N_stacked, 3]."""
        pyr = graphs.pyramid
        b, s = parts[0].shape[:2]
        flat = [p.flatten(0, 1) for p in parts]
        nodes = [pyr.pyramid[i].node_algebra(flat[:len(pyr.pyramid[i].levels)])
                 for i in range(pyr.num_levels)]
        return torch.cat(nodes, 1).unflatten(0, (b, s))

    # ---------------------------------------------------------------- action
    def act(self, graphs: WorldGraphs, z: Latent, omega_node: Tensor,
            terms: Optional[CompositeTerms] = None) -> Latent:
        """Phi_{exp(dt omega)}(z), level by level with that level's cap; with
        composite `terms`, the split step of the composite group around it."""
        if terms is None:
            return self.rotate(graphs, z, omega_node)
        c = self.composite

        def rotate(h, v):
            out = self.rotate(graphs, Latent(h, v), omega_node)
            return out.h, out.v

        shift = None if terms.shift is None else (
            lambda h, v, dt: vert_shift(h, v, terms.shift, dt, c.max_shift))
        fibre = lambda h, v, dt: self.fibre.flow(h, v, terms.coef, dt)
        with torch.autocast(omega_node.device.type, enabled=False):
            h, v = split_step(z.h.float(), z.v.float(), rotate, shift, fibre,
                              self.cfg.dt, c.splitting)
        return Latent(h, v)

    def rotate(self, graphs: WorldGraphs, z: Latent, omega_node: Tensor) -> Latent:
        """The SO(3) part: every level moved by its blended scale chain."""
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
