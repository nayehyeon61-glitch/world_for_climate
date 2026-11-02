"""GraphCast training loss: latitude- and pressure-level-weighted MSE.

Ported from `weathernext/utils/losses.py` (`weighted_mse_per_level`,
`normalized_latitude_weights`, `normalized_level_weights`) and the
per-variable weights of `graphcast.GraphCast.loss_and_predictions`:

    loss = sum_v w_v * mean_{lat, lon, level}( a(lat) * l(level) * err^2 )

with a(lat) ~ cos(lat) (unit mean, special pole cells) and
l(level) = level / mean(levels). On flat channel tensors this is a weighted sum
over channels with weight w_v * l(level) / num_levels(v).
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np
import torch
from torch import Tensor, nn

from ..data.graphcast_mapping import GRAPHCAST_PER_VARIABLE_WEIGHTS, ChannelLayout


def _uniform_delta(latitude: np.ndarray) -> float:
    diff = np.diff(latitude)
    if not np.allclose(diff[0], diff):
        raise ValueError("Latitudes are not uniformly spaced.")
    return float(abs(diff[0]))


def normalized_latitude_weights(latitude: np.ndarray) -> np.ndarray:
    """Unit-mean weights proportional to the area of each latitude band."""
    latitude = np.asarray(latitude, np.float64)
    delta = _uniform_delta(latitude)
    if np.any(np.isclose(np.abs(latitude), 90.)):
        if not (np.isclose(latitude.max(), 90.) and np.isclose(latitude.min(), -90.)):
            raise ValueError("Latitudes with poles must span [-90, 90].")
        weights = np.cos(np.deg2rad(latitude)) * np.sin(np.deg2rad(delta / 2))
        pole = np.isclose(np.abs(latitude), 90.)
        weights[pole] = np.sin(np.deg2rad(delta / 4)) ** 2
    else:
        if not (np.isclose(latitude.max(), 90 - delta / 2)
                and np.isclose(latitude.min(), -90 + delta / 2)):
            raise ValueError("Latitudes must start/end at +-(90 - delta/2).")
        weights = np.cos(np.deg2rad(latitude))
    return weights / weights.mean()


def graphcast_channel_weights(
        layout: ChannelLayout,
        per_variable_weights: Optional[Mapping[str, float]] = None) -> np.ndarray:
    """Weights [C_tgt] reproducing GraphCast's per-variable / per-level loss."""
    per_variable_weights = (GRAPHCAST_PER_VARIABLE_WEIGHTS
                            if per_variable_weights is None else per_variable_weights)
    levels = np.asarray(layout.task.pressure_levels, np.float64)
    mean_level = levels.mean()
    weights = []
    for c in layout.targets:
        w = per_variable_weights.get(c.name, 1.0)
        if c.level is not None:
            w *= (c.level / mean_level) / len(levels)
        weights.append(w)
    return np.asarray(weights, np.float32)


class GraphCastLoss(nn.Module):
    """Weighted MSE on normalised target tensors [..., C_tgt, H, W]."""

    def __init__(self, layout: ChannelLayout, latitude: np.ndarray,
                 per_variable_weights: Optional[Mapping[str, float]] = None):
        super().__init__()
        self.register_buffer("lat_w", torch.tensor(
            normalized_latitude_weights(latitude), dtype=torch.float32))
        self.register_buffer("channel_w", torch.tensor(
            graphcast_channel_weights(layout, per_variable_weights)))

    def per_channel(self, pred: Tensor, target: Tensor) -> Tensor:
        err = (pred.float() - target.float()).square() * self.lat_w[:, None]
        return err.mean((-2, -1))  # [..., C]

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return (self.per_channel(pred, target) * self.channel_w).sum(-1).mean()
