"""Probabilistic metrics for ensembles from the stochastic transition (H3).

GraphCast is deterministic, so these are additional to the GraphCast
protocol (GenCast-style): area-weighted CRPS (fair estimator), ensemble-mean
RMSE and the spread / skill ratio.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
from torch import Tensor

from ..data.graphcast_mapping import Channel
from ..train.losses import normalized_latitude_weights
from .graphcast_metrics import weighted_mean


def crps_fair(ens: Tensor, target: Tensor) -> Tensor:
    """Fair CRPS per point, E|X - y| - E|X - X'| / 2 with the unbiased pair
    mean, via the sorted-ensemble identity (no M x M pair tensor).
    ens [M, ...], target [...]."""
    m = ens.shape[0]
    skill = (ens - target[None]).abs().mean(0)
    if m == 1:
        return skill
    rank = torch.arange(1, m + 1, dtype=ens.dtype).reshape((m,) + (1,) * (ens.ndim - 1))
    pair_mean = 2 * ((2 * rank - m - 1) * ens.sort(0).values).sum(0) / (m * (m - 1))
    return skill - 0.5 * pair_mean


class EnsembleMetrics:
    """Accumulates CRPS, ensemble-mean RMSE and spread over initialisations."""

    def __init__(self, channels: Sequence[Channel], latitude: np.ndarray,
                 lead_hours: Sequence[int]):
        self.channels = list(channels)
        self.lead_hours = list(lead_hours)
        self.lat_w = torch.tensor(normalized_latitude_weights(latitude),
                                  dtype=torch.float32)
        shape = (len(self.lead_hours), len(self.channels))
        self.sum_crps = torch.zeros(shape, dtype=torch.float64)
        self.sum_mse = torch.zeros(shape, dtype=torch.float64)
        self.sum_var = torch.zeros(shape, dtype=torch.float64)
        self.count = 0
        self.members = None

    @torch.no_grad()
    def update(self, ens: Tensor, target: Tensor) -> None:
        """ens [M, B, L, C, H, W], target [B, L, C, H, W] (physical units)."""
        ens, target = ens.float().cpu(), target.float().cpu()
        self.members = ens.shape[0]
        crps = weighted_mean(crps_fair(ens, target), self.lat_w)
        mean = ens.mean(0)
        mse = weighted_mean((mean - target).square(), self.lat_w)
        var = weighted_mean(ens.var(0, unbiased=ens.shape[0] > 1), self.lat_w)
        self.sum_crps += crps.sum(0).double()
        self.sum_mse += mse.sum(0).double()
        self.sum_var += var.sum(0).double()
        self.count += target.shape[0]

    def compute(self) -> Dict[str, np.ndarray]:
        m = self.members or 1
        rmse = (self.sum_mse / self.count).sqrt()
        spread = (self.sum_var / self.count).sqrt() * np.sqrt((m + 1) / m)
        return {"crps": (self.sum_crps / self.count).numpy(),
                "ens_mean_rmse": rmse.numpy(),
                "spread_skill": (spread / rmse.clamp_min(1e-12)).numpy()}
