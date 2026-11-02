"""Per-channel normalisation statistics (GraphCast's mean/stddev/diffs_stddev).

Scalars are normalised as (x - mean) / std like GraphCast's
`normalization.InputsAndResiduals`. Wind components are tangent vectors, so
each (u, v) pair gets zero location and one shared scale (the RMS component):
subtracting a mean wind or rescaling u and v differently would change the
direction of the vector and break rotation equivariance of the encoder.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Dict, Iterable, Sequence

import numpy as np

from .graphcast_mapping import VECTOR_PAIRS, Channel


@dataclasses.dataclass
class NormalizationStats:
    mean: Dict[str, float]
    std: Dict[str, float]
    diff_std: Dict[str, float]

    def arrays(self, channels: Sequence[Channel]):
        mean = np.array([self.mean[c.key] for c in channels], np.float32)
        std = np.array([self.std[c.key] for c in channels], np.float32)
        return mean, std

    def normalize(self, x: np.ndarray, channels: Sequence[Channel]) -> np.ndarray:
        """x [..., C, H, W] in physical units -> normalised."""
        mean, std = self.arrays(channels)
        return (x - mean[:, None, None]) / std[:, None, None]

    def unnormalize(self, x, channels: Sequence[Channel]):
        """Works for numpy arrays and torch tensors of shape [..., C, H, W]."""
        mean, std = self.arrays(channels)
        if not isinstance(x, np.ndarray):
            import torch
            mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype)
            std = torch.as_tensor(std, device=x.device, dtype=x.dtype)
        return x * std[:, None, None] + mean[:, None, None]

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=1)

    @classmethod
    def load(cls, path: str) -> "NormalizationStats":
        with open(path) as f:
            return cls(**json.load(f))


class _Moments:
    def __init__(self):
        self.n = 0
        self.s1 = None
        self.s2 = None

    def add(self, x: np.ndarray, axes) -> None:
        x = x.astype(np.float64)
        s1, s2 = x.sum(axes), (x * x).sum(axes)
        self.n += int(np.prod([x.shape[a] for a in axes]))
        self.s1 = s1 if self.s1 is None else self.s1 + s1
        self.s2 = s2 if self.s2 is None else self.s2 + s2

    def mean_std(self):
        mean = self.s1 / self.n
        var = np.maximum(self.s2 / self.n - mean ** 2, 0.)
        return mean, np.sqrt(var), self.s2 / self.n


class _RawWindows:
    """Picklable view returning raw windows, for parallel reading."""

    def __init__(self, dataset, indices):
        self.dataset, self.indices = dataset, list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        frames, forc, _ = self.dataset.raw_window(self.indices[i])
        return frames, forc


def compute_stats(dataset, indices: Iterable[int], num_workers: int = 0,
                  log=None) -> NormalizationStats:
    """Streaming statistics from a `GraphCastWindowDataset` (unnormalised).

    Uses target channels (a superset of the dynamic inputs), forcings and the
    static fields. `diff_std` is the std of 6h differences. With
    `num_workers > 0` the windows are read by DataLoader workers (I/O bound on
    the 0.25 degree archive); the moments are accumulated in this process.
    """
    layout = dataset.layout
    frames_m, diff_m, forc_m = _Moments(), _Moments(), _Moments()
    windows = _RawWindows(dataset, indices)
    if num_workers > 0:
        import torch
        iterator = torch.utils.data.DataLoader(
            windows, batch_size=None, num_workers=num_workers,
            collate_fn=lambda x: x, prefetch_factor=2)
    else:
        iterator = (windows[i] for i in range(len(windows)))
    for n, (frames, forc) in enumerate(iterator):
        frames, forc = np.asarray(frames), np.asarray(forc)
        frames_m.add(frames, (0, 2, 3))
        diff_m.add(np.diff(frames, axis=0), (0, 2, 3))
        forc_m.add(forc, (0, 2, 3))
        if log is not None and (n + 1) % 50 == 0:
            log(f"[stats] {n + 1}/{len(windows)} windows")
    static_m = _Moments()
    static_m.add(dataset.static[None], (0, 2, 3))

    mean, std, diff_std = {}, {}, {}
    for moments, channels in ((frames_m, layout.targets),
                              (forc_m, layout.forcings),
                              (static_m, layout.static)):
        m, s, second = moments.mean_std()
        for i, c in enumerate(channels):
            mean[c.key], std[c.key] = float(m[i]), float(max(s[i], 1e-8))
        if channels is layout.targets:
            _, ds, _ = diff_m.mean_std()
            for i, c in enumerate(channels):
                diff_std[c.key] = float(max(ds[i], 1e-8))
            index = {c: i for i, c in enumerate(channels)}
            for u_name, v_name in VECTOR_PAIRS:
                for c in channels:
                    partner = Channel(v_name, c.level)
                    if c.name == u_name and partner in index:
                        rms = float(np.sqrt(0.5 * (second[index[c]] + second[index[partner]])))
                        for key in (c.key, partner.key):
                            mean[key], std[key] = 0.0, max(rms, 1e-8)
    return NormalizationStats(mean=mean, std=std, diff_std=diff_std)
