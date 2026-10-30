"""Lie generator and structural transition (research design 6.B / 6.C).

    B. xi_t = f_theta(Z_{t-k+1:t}, c_{t+1})  in  g = so(3)^{shells x patches}
    C. Zbar_{t+1} = Phi_{exp(dt xi_t)}(Z_t)

f_theta is a Shell-GNN on the latent mesh that proposes a tangent velocity
field u_i = sum_c alpha_ic V_ic (equivariant: alpha are invariant scalars, V
are the latent tangent vectors) and a spin rate s_i. Per (shell, patch) the
velocity is projected onto the Lie algebra by a least-squares Killing-field
fit, omega_P = argmin |omega x p - u|^2, plus spin s_P along the patch centre.
Node elements omega_i are blended in the algebra and exponentiated per node;
`SphereRotationAction` then transports scalars and vectors (semi-Lagrangian).

Shell coupling ("liquid centipede" hypothesis):
  * `independent`: every shell gets its own omega_s = delta_s.
  * `chain`: shells form an articulated chain anchored at the surface, like the
    modules of a centipede robot joined by 3-DOF flexible joints. The pose of
    shell s is the pose of shell s-1 composed with a joint rotation,
        g_s = g_{s-1} exp(gate_s * delta_s)  (forward kinematics on SO(3)),
    computed exactly as omega_s = log(exp(omega_{s-1}) exp(gate_s delta_s)).
    The joint rotation is the vertical shear between shells; the learned gate
    in (0, 1) switches between rigid (barotropic, gate -> 0) and flexible
    (baroclinic, gate -> 1) coupling, and |gate_s delta_s|^2 is returned for a
    torsion-spring (joint stiffness) penalty.

What the Lie part can represent: locally rigid advection / rotation of the
latent state with vertical shear (one omega per shell). Divergence,
deformation, sources and sinks (heating, condensation) and topology changes
(splitting / merging of systems, as for the liquid robot) cannot be written
as an isometry and are left to the residual / stochastic transition.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence, Tuple

import torch
from torch import Tensor, nn

from ..geometry.group_actions import clamp_rotation, so3_exp, so3_log
from ..geometry.spherical_mesh import icosphere_hierarchy, max_edge_length
from .graphs import WorldGraphs
from .latent import Latent, LatentEmbedding
from .layers import ShellProcessor


@dataclasses.dataclass
class LieConfig:
    layers: int = 2
    max_rotation_edges: float = 1.5  # |omega dt| cap in latent-mesh edge lengths
    dt: float = 1.0  # one GraphCast step (6h) per transition
    shell_coupling: str = "independent"  # "independent" | "chain"


def compose(a: Tensor, b: Tensor) -> Tensor:
    """log(exp(a) exp(b)) in so(3) (= a + b + [a, b] / 2 + ... by BCH)."""
    return so3_log(so3_exp(a) @ so3_exp(b))


class LieGenerator(nn.Module):

    def __init__(self, graphs: WorldGraphs, num_shells: int, history: int,
                 forcing_dim: int, cs: int, cv_latent: int, d: int, cv: int,
                 hidden_layers: int, cfg: LieConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = LatentEmbedding(history, cs, cv_latent, forcing_dim,
                                     graphs.struct_dim, num_shells, d, cv,
                                     hidden_layers)
        self.processor = ShellProcessor(d, cv, graphs.latent_mesh.edge_dim,
                                        cfg.layers, num_shells, hidden_layers)
        if cfg.shell_coupling not in ("independent", "chain"):
            raise ValueError(f"unknown shell_coupling {cfg.shell_coupling}")
        self.velocity = nn.Linear(d, cv)
        self.spin = nn.Linear(d, 1)
        self.joint_gate = nn.Linear(d, 1) if cfg.shell_coupling == "chain" else None
        for lin in (self.velocity, self.spin):  # start at the identity action
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        latent_mesh = icosphere_hierarchy(graphs.cfg.latent_splits)[-1]
        self.max_angle = cfg.max_rotation_edges * max_edge_length(latent_mesh)

    def forward(self, graphs: WorldGraphs, history: Sequence[Latent],
                node_forcing: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns omega_patch [B, S, P, 3], omega_node [B, S, N, 3] and the
        joint rotations [B, S, P, 3] (= omega_patch for independent shells)."""
        return self.algebra(graphs, *self.features(graphs, history, node_forcing))

    def features(self, graphs: WorldGraphs, history: Sequence[Latent],
                 node_forcing: Tensor) -> Tuple[Tensor, Tensor]:
        """Processed node features (h [B, S, N, d], v [B, S, N, cv, 3]) that the
        algebra head reads; the theory programmer adds its physics head here."""
        h, v = self.embed(history, node_forcing, graphs.latent_struct)
        return self.processor(graphs.latent_mesh, h, v)

    def algebra(self, graphs: WorldGraphs, h: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        with torch.autocast(h.device.type, enabled=False):  # so(3) algebra in fp32
            return self._algebra(graphs, h.float(), v.float())

    def _algebra(self, graphs: WorldGraphs, h: Tensor, v: Tensor):
        b, s, n = h.shape[:3]
        u = (self.velocity(h)[..., None] * v).sum(-2)  # tangent velocity
        patches = graphs.patches
        delta = patches.killing_projection(u.flatten(0, 1))
        spin = patches.pool(self.spin(h).flatten(0, 1))
        delta = delta + spin * patches.centres[None]
        delta = delta.unflatten(0, (b, s))
        if self.joint_gate is None:
            joints = omega = delta
        else:
            gate = torch.sigmoid(self.joint_gate(patches.pool(h.flatten(0, 1))))
            gate = gate.unflatten(0, (b, s))
            gate = torch.cat([torch.ones_like(gate[:, :1]), gate[:, 1:]], 1)
            joints = gate * delta
            chain = [joints[:, 0]]
            for k in range(1, s):
                chain.append(compose(chain[-1], joints[:, k]))
            omega = torch.stack(chain, 1)
        omega_node = patches.blend(omega.flatten(0, 1)).unflatten(0, (b, s))
        return omega, omega_node, joints

    def act(self, graphs: WorldGraphs, z: Latent, omega_node: Tensor) -> Latent:
        """Phi_{exp(dt omega)}(z).

        |dt omega| is smoothly limited to `max_angle` (a CFL-like limiter that
        keeps departure points inside the triangles searched by the action).
        """
        with torch.autocast(omega_node.device.type, enabled=False):  # exp map + interpolation in fp32
            step = clamp_rotation(self.cfg.dt * omega_node.float(), self.max_angle)
            rot = so3_exp(step).flatten(0, 1)
            h, v = graphs.rotation_action(z.h.float().flatten(0, 1),
                                          z.v.float().flatten(0, 1), rot)
        b, s = z.h.shape[:2]
        return Latent(h.unflatten(0, (b, s)), v.unflatten(0, (b, s)))
