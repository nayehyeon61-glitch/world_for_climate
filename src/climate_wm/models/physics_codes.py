"""Column-local residual physics primitives for the theory program (stage 3, phase 2).

Numerical weather prediction splits every step into a dynamical core
(adiabatic transport) and physics parameterisations (radiation, convection,
boundary layer, ...). The theory program mirrors that split: the Lie code
is the dynamical core (an so(3) transport of the patch), and a *physics
code* is executed afterwards. The physics executor is shaped like a
parameterisation and nothing else:

  * it is column-local: it reads and writes the shells of one latent node
    and never exchanges information horizontally, so it cannot imitate
    transport;
  * `radiation` changes the scalars of a shell as a function of the node's
    forcing (TOA solar radiation, time of day / year);
  * `mixing` moves latent content between adjacent shells in flux form with a
    non-negative, down-gradient coefficient: what leaves shell s enters
    shell s+1, so the column sum is conserved and vertical gradients shrink;
  * `surface` exchanges content between the surface shell and the boundary
    layer shell, conditioned on the node's static fields (land / sea,
    orography) and forcing.

Each code z of the physics codebook belongs to one family (the codebook is
split in three equal blocks) and conditions its family's executor through
its embedding. Code 0 is the identity. All tendencies start at zero (the
final layers are zero-initialised), so the model begins as "Lie only".
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .latent import Latent

FAMILIES = ("radiation", "mixing", "surface")


def _tendency_mlp(inp: int, hidden: int, out: int, layers: int) -> nn.Sequential:
    """MLP without a final LayerNorm whose last layer starts at zero."""
    mods = []
    for _ in range(max(layers, 1)):
        mods += [nn.Linear(inp, hidden), nn.SiLU()]
        inp = hidden
    last = nn.Linear(inp, out)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)


def code_family(idx: Tensor, num_codes: int) -> Tensor:
    """Family index (0..2) of each physics code; -1 for the identity code 0."""
    fam = (idx - 1) * len(FAMILIES) // max(num_codes, 1)
    return torch.where(idx > 0, fam.clamp(0, len(FAMILIES) - 1), torch.full_like(idx, -1))


def family_weights(idx: Tensor, num_codes: int) -> Tensor:
    """One-hot family weights [..., 3]; all zero for the identity code."""
    fam = code_family(idx, num_codes)
    return F.one_hot(fam.clamp_min(0), len(FAMILIES)).float() * (fam >= 0).float()[..., None]


class PhysicsExecutor(nn.Module):
    """Executes blended physics codes on a latent column field."""

    def __init__(self, forcing_dim: int, static_dim: int, cs: int, cv: int,
                 hidden: int, hidden_layers: int, code_dim: int, scale: float):
        super().__init__()
        self.scale = scale
        self.radiation = _tendency_mlp(cs + forcing_dim + code_dim, hidden, cs, hidden_layers)
        self.kappa_h = _tendency_mlp(2 * cs + code_dim, hidden, cs, hidden_layers)
        self.kappa_v = _tendency_mlp(2 * cs + code_dim, hidden, cv, hidden_layers)
        self.surface = _tendency_mlp(2 * cs + forcing_dim + static_dim + code_dim, hidden, cs,
                                     hidden_layers)

    def forward(self, z: Latent, e: Tensor, w: Tensor, nf: Tensor, ns: Tensor) -> Latent:
        """z: latent with h [B, S, N, Cs], v [B, S, N, Cv, 3]; e [B, S, N, dz] code
        embeddings blended onto nodes; w [B, S, N, 3] family weights (0 for the
        identity); nf [B, N, F] node forcing; ns [B, N, St] node static fields."""
        h, v = z.h.float(), z.v.float()
        b, s, n = h.shape[:3]
        nf_s = nf[:, None].expand(b, s, n, nf.shape[-1])
        dh = self.radiation(torch.cat([h, nf_s, e], -1)) * w[..., 0:1]
        dv = torch.zeros_like(v)
        if s > 1:
            lo, hi = h[:, :-1], h[:, 1:]
            inp = torch.cat([lo, hi, e[:, :-1]], -1)
            w_mix = w[:, :-1, :, 1:2]                       # the lower shell's code drives the interface
            fh = F.softplus(self.kappa_h(inp)) * w_mix * (hi - lo)       # [B, S-1, N, Cs]
            fv = (F.softplus(self.kappa_v(inp)) * w_mix)[..., None] * (v[:, 1:] - v[:, :-1])
            dh = dh + F.pad(fh, (0, 0, 0, 0, 0, 1)) - F.pad(fh, (0, 0, 0, 0, 1, 0))
            dv = dv + F.pad(fv, (0, 0, 0, 0, 0, 0, 0, 1)) - F.pad(fv, (0, 0, 0, 0, 0, 0, 1, 0))
            inp0 = torch.cat([h[:, 0], h[:, 1], nf, ns, e[:, 0]], -1)
            f0 = self.surface(inp0) * w[:, 0, :, 2:3]                    # surface -> boundary layer
            dh = dh + torch.cat([-f0[:, None], f0[:, None], torch.zeros_like(dh[:, 2:])], 1)
        return Latent(h + self.scale * dh, v + self.scale * dv)

    def conservation_check(self, dz: Latent) -> Tuple[Tensor, Tensor]:
        """Column sums of a mixing / surface tendency (should vanish)."""
        return dz.h.sum(1).abs().max(), dz.v.sum(1).abs().max()
