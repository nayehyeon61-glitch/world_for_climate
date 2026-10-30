"""Latent state Z on the latent mesh: per (shell, node) scalars + tangent vectors.

Z = (h, V), h [B, S, N, Cs], V [B, S, N, Cv, 3] with V tangent to S^2 at the
node. M_Z is (a discretisation of) the space of sections of the bundle
R^Cs x (T S^2)^Cv over S x S^2. The fibre is a vector space, so the
Riemannian exponential at Zbar is Exp_Zbar(r) = P(Zbar + r) (P = tangent
projection); the non-trivial geometry lives in the base S^2 and in the SO(3)
group action (see geometry/group_actions.py). This is the "chart / tangent
approximation" the research design starts from.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import torch
from torch import Tensor, nn

from ..geometry.tangent_frames import project_tangent
from .layers import MLP, VectorLinear


@dataclasses.dataclass
class Latent:
    h: Tensor  # [B, S, N, Cs]
    v: Tensor  # [B, S, N, Cv, 3]

    @property
    def batch_size(self) -> int:
        return self.h.shape[0]

    def __add__(self, other: "Latent") -> "Latent":
        return Latent(self.h + other.h, self.v + other.v)

    def __sub__(self, other: "Latent") -> "Latent":
        return Latent(self.h - other.h, self.v - other.v)

    def scale(self, factor) -> "Latent":
        return Latent(self.h * factor, self.v * factor)

    def detach(self) -> "Latent":
        return Latent(self.h.detach(), self.v.detach())

    def project(self, pos: Tensor) -> "Latent":
        """Projects vectors to the tangent planes; pos [N, 3]."""
        b, s, n = self.h.shape[:3]
        return Latent(self.h, project_tangent(self.v, pos.expand(b, s, n, 3)))

    def exp(self, residual: "Latent", pos: Tensor) -> "Latent":
        """Exp_Z(r) on the latent bundle (fibre-wise addition + projection)."""
        return (self + residual).project(pos)

    def pack(self) -> Tensor:
        """[B, S, N, Cs + 3 Cv] flat view (used by the diffusion model)."""
        return torch.cat([self.h, self.v.flatten(-2)], -1)

    @staticmethod
    def unpack(x: Tensor, cs: int) -> "Latent":
        return Latent(x[..., :cs], x[..., cs:].unflatten(-1, (-1, 3)))

    def squared_error(self, other: "Latent") -> Tensor:
        """Mean squared difference over all latent coordinates (scalar)."""
        return torch.cat([(self.h - other.h).flatten(1),
                          (self.v - other.v).flatten(1)], 1).square().mean()


def _per_sample(x: Tensor, factor: Tensor) -> Tensor:
    return x * factor.reshape(factor.shape + (1,) * (x.ndim - factor.ndim))


def scale_per_sample(z: Latent, factor: Tensor) -> Latent:
    """Multiplies every sample b of z by factor[b] (factor [B])."""
    return Latent(_per_sample(z.h, factor), _per_sample(z.v, factor))


class LatentEmbedding(nn.Module):
    """Embeds latents (+ node forcings + node structure) into (h, v) of width
    (d, cv) for the transition networks; adds a learned shell embedding."""

    def __init__(self, num_latents: int, cs: int, cv_latent: int,
                 forcing_dim: int, struct_dim: int, num_shells: int, d: int,
                 cv: int, hidden_layers: int = 1):
        super().__init__()
        self.mlp = MLP(num_latents * cs + forcing_dim + struct_dim, d, d,
                       hidden_layers)
        self.vec = VectorLinear(num_latents * cv_latent, cv)
        self.shell_embed = nn.Parameter(0.02 * torch.randn(num_shells, d))

    def forward(self, latents: Sequence[Latent], node_forcing: Tensor,
                struct: Tensor):
        """node_forcing [B, N, F], struct [N, F_s] -> h [B,S,N,d], v [B,S,N,cv,3]."""
        b, s, n = latents[0].h.shape[:3]
        feats = [z.h for z in latents]
        feats.append(node_forcing[:, None].expand(b, s, n, -1))
        feats.append(struct.expand(b, s, n, -1))
        h = self.mlp(torch.cat(feats, -1)) + self.shell_embed[None, :, None]
        v = self.vec(torch.cat([z.v for z in latents], -2))
        return h, v
