"""Stage 13 E2 / E3: a learned corrector of the column tendencies with an energy constraint.

감자's idea (10:28 KST 11/01): control the blow-ups with an energy constraint
through a neural operator. E1 (physics_world_model.energy_bound) clamps the
column's energy input and removes the excess uniformly per unit mass; it fixes
*where* the energy is taken from. E2 learns that: a spherical Fourier neural
operator (SFNO, as in stage 11) reads the column's input view of the state and
the column tendencies (Q, S, F_u, F_v after the budget and the E1 bound) and
returns a correction of the four tendencies whose output is projected to be

    energy neutral    c_p int dQ dp/g + L_v int dS dp/g = 0   (per column)
    momentum neutral  int dF dp = 0
    no work           int dF . u dp <= 0

so the corrector can only redistribute heat, moisture and momentum inside a
column (vertically, and between heat and moisture); the bounded energy input of
E1 is kept. It is trained by the free-run forecast loss of the frozen model on
2018-19 (train/memory_filter.train_filter with new_prefix "corrector.").

E3 (`neutral: false`, `soft_weight > 0`) is the soft control: no projection, and
the training loss gets soft_weight * mean(relu(|flux| - soft_bound) / soft_bound)^2
over the column calls (the total flux after the correction).

`kind: mlp` is a per-column corrector (the same inputs through a 1 x 1
network): the local control for the non-local operator.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import torch
from torch import Tensor, nn

from .primitive_core import CP

GRAV = 9.80665
L_V = 2.501e6  # J/kg
DAY = 86400.0
SCALES = {"Q": 10.0, "S": 2e-3, "F": 10.0}  # physics_world_model.SCALES (per day units)


@dataclasses.dataclass
class CorrectorConfig:
    enabled: bool = False
    kind: str = "sfno"            # sfno (non-local; neuraloperator) | mlp (per column)
    hidden: int = 16
    layers: int = 2
    modes: int = 16               # SFNO modes (degree and order), capped at the core truncation + 1
    inputs: str = "state_tendency"  # state_tendency | tendency (the corrector sees only the column output)
    neutral: bool = True          # E2: the correction is projected (energy, momentum, work)
    soft_weight: float = 0.0      # E3: penalty weight on the total flux beyond soft_bound
    soft_bound: float = 500.0     # W/m^2

    def __post_init__(self):
        if self.kind not in ("sfno", "mlp"):
            raise ValueError("corrector kind must be sfno or mlp")
        if self.inputs not in ("state_tendency", "tendency"):
            raise ValueError("corrector inputs must be state_tendency or tendency")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "CorrectorConfig":
        return cls(**(d or {}))


def _mlp2d(n_in: int, n_out: int, hidden: int, layers: int) -> nn.Module:
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Conv2d(d, hidden, 1), nn.SiLU()]
        d = hidden
    last = nn.Conv2d(d, n_out, 1)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)


class EnergyCorrector(nn.Module):
    def __init__(self, cfg: CorrectorConfig, n_levels: int, n_state: int, trunc: int):
        super().__init__()
        self.cfg, self.n = cfg, n_levels
        n_in = (n_state if cfg.inputs == "state_tendency" else 0) + 4 * n_levels
        n_out = 4 * n_levels
        if cfg.kind == "sfno":
            from .operator_control import _sfno
            self.net = _sfno(n_in, n_out, cfg.hidden, cfg.layers, min(cfg.modes, trunc + 1))
        else:
            self.net = _mlp2d(n_in, n_out, cfg.hidden, cfg.layers)
        self.penalties: Optional[list] = None  # E3: set to a list to collect the soft penalty per call
        self.track: Optional[list] = None      # diagnostics: per call [B, 2] RMS of dQ (K/day) and |dF| (m/s/day)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        out = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        for key in [k for k in out if k.endswith("_metadata")]:  # neuralop adds a tensor-less entry
            del out[key]
        return out

    def correction(self, out: Dict[str, Tensor], x: Tensor, u: Tensor, v: Tensor, dp_layer: Tensor):
        """The projected correction (dQ, dS, dF_u, dF_v) in physical units per second."""
        n, per_s = self.n, 1.0 / DAY
        tend = torch.cat([out["Q"] / (SCALES["Q"] * per_s), out["S"] / (SCALES["S"] * per_s),
                          out["F_u"] / (SCALES["F"] * per_s), out["F_v"] / (SCALES["F"] * per_s)], 1)
        inp = torch.cat([x.permute(0, 3, 1, 2), tend], 1) if self.cfg.inputs == "state_tendency" else tend
        y = self.net(inp.contiguous())
        dq = y[:, :n] * SCALES["Q"] * per_s
        ds = y[:, n:2 * n] * SCALES["S"] * per_s
        dfu = y[:, 2 * n:3 * n] * SCALES["F"] * per_s
        dfv = y[:, 3 * n:] * SCALES["F"] * per_s
        if self.cfg.neutral:
            dp = dp_layer[None, :, None, None]
            mass = dp_layer.sum() / GRAV
            e = (CP * dq * dp).sum(1) / GRAV + (L_V * ds * dp).sum(1) / GRAV
            dq = dq - (e / (CP * mass))[:, None]
            w = (dp_layer / dp_layer.sum())[None, :, None, None]
            dfu = dfu - (dfu * w).sum(1, keepdim=True)
            dfv = dfv - (dfv * w).sum(1, keepdim=True)
            # no work: remove the component along the column-mean-free wind (keeps int dF dp = 0;
            # a uniform wind column does no work with a momentum-neutral dF)
            up = u - (u * w).sum(1, keepdim=True)
            vp = v - (v * w).sum(1, keepdim=True)
            work = ((dfu * u + dfv * v) * dp).sum(1) / GRAV
            ke2 = ((up * up + vp * vp) * dp).sum(1) / GRAV
            lam = torch.relu(work) / ke2.clamp_min(1.0)
            dfu = dfu - lam[:, None] * up
            dfv = dfv - lam[:, None] * vp
        return dq, ds, dfu, dfv

    def forward(self, out: Dict[str, Tensor], x: Tensor, u: Tensor, v: Tensor, dp_layer: Tensor) -> Dict[str, Tensor]:
        dq, ds, dfu, dfv = self.correction(out, x, u, v, dp_layer)
        out = dict(out)
        out["Q"] = out["Q"] + dq
        out["S"] = out["S"] + ds
        out["F_u"] = out["F_u"] + dfu
        out["F_v"] = out["F_v"] + dfv
        if self.cfg.soft_weight > 0 and self.penalties is not None:
            dp = dp_layer[None, :, None, None]
            flux = (CP * out["Q"] * dp).sum(1) / GRAV + (L_V * out["S"] * dp).sum(1) / GRAV
            excess = torch.relu(flux.abs() - self.cfg.soft_bound) / self.cfg.soft_bound
            self.penalties.append(self.cfg.soft_weight * (excess ** 2).mean())
        if self.track is not None:
            self.track.append(torch.stack([(dq * DAY).pow(2).mean((1, 2, 3)).sqrt(),
                                           ((dfu ** 2 + dfv ** 2) * DAY ** 2).mean((1, 2, 3)).sqrt()], 1)
                              .detach().cpu())
        return out
