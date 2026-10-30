"""GraphCast evaluation protocol (Lam et al. 2023, Supplement 5.3).

All metrics are computed in float32 in the native physical units of every
variable (no normalisation), against the model's ground truth (ERA5):

  RMSE(j, tau)     = 1/|D| sum_{d0} sqrt( 1/|G| sum_i a_i (xhat - x)^2 )     (20)
  RMSE_trad(j,tau) = sqrt( 1/|D| sum_{d0} 1/|G| sum_i a_i (xhat - x)^2 )     (21)
  ACC(j, tau)      = 1/|D| sum_{d0} sum_i a_i (xhat-C)(x-C)
                     / sqrt( sum_i a_i (xhat-C)^2  sum_i a_i (x-C)^2 )        (29)
  RMSE skill score = (RMSE_A - RMSE_B) / RMSE_B
  ACC skill score  = (ACC_A - ACC_B) / (1 - ACC_B)

a_i is the cell area normalised to unit mean (latitude weights, see
train/losses.py); C is the climatological mean for the day-of-year (and hour)
of the validity time (GraphCast uses ERA5 1993-2016).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..data.graphcast_mapping import Channel
from ..train.losses import normalized_latitude_weights

# Headline variables of the GraphCast paper / WeatherBench scorecards.
HEADLINE_KEYS = (
    "geopotential@500",
    "temperature@850",
    "specific_humidity@700",
    "u_component_of_wind@850",
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
)


def weighted_mean(x: Tensor, lat_w: Tensor) -> Tensor:
    """Area-weighted mean over the last two (lat, lon) axes."""
    return (x * lat_w[:, None]).mean((-2, -1))


def rmse_per_init(pred: Tensor, target: Tensor, lat_w: Tensor) -> Tensor:
    return weighted_mean((pred - target).square(), lat_w).sqrt()


def acc_per_init(pred: Tensor, target: Tensor, clim: Tensor, lat_w: Tensor) -> Tensor:
    fa, oa = pred - clim, target - clim
    num = weighted_mean(fa * oa, lat_w)
    den = (weighted_mean(fa.square(), lat_w) * weighted_mean(oa.square(), lat_w)).sqrt()
    return num / den.clamp_min(1e-12)


def rmse_skill_score(rmse_a, rmse_b):
    return (rmse_a - rmse_b) / rmse_b


def acc_skill_score(acc_a, acc_b):
    return (acc_a - acc_b) / (1 - acc_b)


class GraphCastMetrics:
    """Streaming accumulator over forecast initialisations.

    update() takes physical-unit tensors [B, L, C, H, W] (B initialisations,
    L lead times) and, optionally, the climatology at the validity times.
    """

    def __init__(self, channels: Sequence[Channel], latitude: np.ndarray,
                 lead_hours: Sequence[int], track_bias: bool = False):
        self.channels = list(channels)
        self.track_bias = track_bias  # [L, C, H, W] float64: small grids only
        self.lead_hours = list(lead_hours)
        self.lat_w = torch.tensor(normalized_latitude_weights(latitude),
                                  dtype=torch.float32)
        shape = (len(self.lead_hours), len(self.channels))
        self.sum_rmse = torch.zeros(shape, dtype=torch.float64)
        self.sum_mse = torch.zeros(shape, dtype=torch.float64)
        self.sum_acc = torch.zeros(shape, dtype=torch.float64)
        self.sum_bias = None  # per-location mean bias error (eq. 26)
        self.count = 0
        self.count_acc = 0

    @torch.no_grad()
    def update(self, pred: Tensor, target: Tensor,
               clim: Optional[Tensor] = None) -> None:
        pred, target = pred.float().cpu(), target.float().cpu()
        lat_w = self.lat_w
        mse = weighted_mean((pred - target).square(), lat_w)  # [B, L, C]
        self.sum_mse += mse.sum(0).double()
        self.sum_rmse += mse.sqrt().sum(0).double()
        if self.track_bias:
            bias = (pred - target).sum(0).double()
            self.sum_bias = bias if self.sum_bias is None else self.sum_bias + bias
        self.count += pred.shape[0]
        if clim is not None:
            acc = acc_per_init(pred, target, clim.float().cpu(), lat_w)
            self.sum_acc += acc.sum(0).double()
            self.count_acc += pred.shape[0]

    def compute(self) -> Dict[str, np.ndarray]:
        out = {
            "rmse": (self.sum_rmse / self.count).numpy(),
            "rmse_trad": (self.sum_mse / self.count).sqrt().numpy(),
        }
        if self.count_acc:
            out["acc"] = (self.sum_acc / self.count_acc).numpy()
        if self.sum_bias is not None:
            mbe = self.sum_bias / self.count  # [L, C, H, W]
            w = self.lat_w.double()[:, None]
            out["rms_mbe"] = ((mbe.square() * w).mean((-2, -1))).sqrt().numpy()
        return out

    def table(self, keys: Optional[Sequence[str]] = None) -> List[dict]:
        """Rows {variable, lead_h, rmse, rmse_trad, acc?} for `keys`."""
        res = self.compute()
        index = {c.key: i for i, c in enumerate(self.channels)}
        keys = [k for k in (keys or index) if k in index]
        rows = []
        for k in keys:
            for li, lead in enumerate(self.lead_hours):
                row = {"variable": k, "lead_h": lead,
                       "rmse": float(res["rmse"][li, index[k]]),
                       "rmse_trad": float(res["rmse_trad"][li, index[k]])}
                if "acc" in res:
                    row["acc"] = float(res["acc"][li, index[k]])
                rows.append(row)
        return rows


def format_scorecard(results: Dict[str, GraphCastMetrics], model: str,
                     baseline: str, keys: Sequence[str] = HEADLINE_KEYS) -> str:
    """Text scorecard: RMSE / ACC of every entry and the RMSE skill score of
    `model` against `baseline` (negative = model better)."""
    computed = {name: m.compute() for name, m in results.items()}
    ref = results[model]
    index = {c.key: i for i, c in enumerate(ref.channels)}
    keys = [k for k in keys if k in index]
    with_acc = [n for n in results if "acc" in computed[n]]
    header = f"{'variable':<26}{'lead':>6}"
    header += "".join(f"{'RMSE ' + n:>20}" for n in results)
    header += "".join(f"{'ACC ' + n:>18}" for n in with_acc)
    header += f"{'skill ' + model + '/' + baseline:>26}"
    lines = [header]
    for k in keys:
        j = index[k]
        for li, lead in enumerate(ref.lead_hours):
            line = f"{k:<26}{lead:>5}h"
            line += "".join(f"{computed[n]['rmse'][li, j]:>20.4g}" for n in results)
            line += "".join(f"{computed[n]['acc'][li, j]:>18.3f}" for n in with_acc)
            score = rmse_skill_score(computed[model]["rmse"][li, j],
                                     computed[baseline]["rmse"][li, j])
            line += f"{score:>+26.3f}"
            lines.append(line)
    return "\n".join(lines)
