"""Climatology C_{j,i}^{d0+tau} for the ACC (GraphCast eq. 29).

GraphCast uses the ERA5 1993-2016 mean per variable, level, location and
day-of-year of the validity time. Two sources are supported:

  * `Climatology.from_window_dataset`: computed from our own windows
    (day-of-year x hour-of-day means, optional +-window smoothing); fine for
    smoke tests and coarse grids.
  * `Climatology.from_weatherbench2`: a WeatherBench-2 climatology zarr
    (dims hour, dayofyear, level, latitude, longitude), read lazily; this is
    the practical choice at 0.25 degrees.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..data.graphcast_mapping import Channel


class Climatology:

    def __init__(self, values: Optional[np.ndarray] = None,
                 hours: Sequence[int] = (0, 6, 12, 18), lazy=None,
                 channels: Optional[Sequence[Channel]] = None,
                 coarsener=None, cache_size: int = 64):
        """values [366, len(hours), C, H, W] (NaN for missing days) or a lazy
        xarray dataset in WeatherBench-2 format (frames are read on demand,
        optionally coarsened, and kept in a small LRU cache)."""
        self.values = values
        self.hours = list(hours)
        self.lazy = lazy
        self.channels = list(channels) if channels is not None else None
        self.coarsener = coarsener
        self.cache_size = cache_size
        self._cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

    @classmethod
    def from_window_dataset(cls, dataset, indices: Sequence[int],
                            window_days: int = 0) -> "Climatology":
        sums, counts = None, np.zeros((366, 4))
        hours = (0, 6, 12, 18)
        for k in indices:
            frames, _, times = dataset.raw_window(k)
            for frame, t in zip(frames, times):
                ts = pd.Timestamp(t)
                if ts.hour not in hours:
                    continue
                if sums is None:
                    sums = np.zeros((366, 4) + frame.shape, np.float64)
                d, h = ts.dayofyear - 1, hours.index(ts.hour)
                sums[d, h] += frame
                counts[d, h] += 1
        if window_days:
            kernel = range(-window_days, window_days + 1)
            sums = sum(np.roll(sums, k, axis=0) for k in kernel)
            counts = sum(np.roll(counts, k, axis=0) for k in kernel)
        with np.errstate(invalid="ignore"):
            values = sums / counts[..., None, None, None]
        values[counts == 0] = np.nan
        return cls(values.astype(np.float32), hours, channels=dataset.layout.targets)

    @classmethod
    def from_weatherbench2(cls, path: str, channels: Sequence[Channel],
                           load: bool = True, coarsen: int = 1) -> "Climatology":
        """WeatherBench-2 climatology zarr (local or gs://). load=True reads the
        selected channels into memory ([366, hours, C, H, W]; ~1 GB at 64x32 x
        83 ch); load=False reads frames lazily, e.g. the 0.25 deg climatology
        coarsened on the fly (`coarsen=4` for a 1 deg model)."""
        import xarray as xr
        opts = {"storage_options": {"token": "anon"}} if path.startswith("gs://") else {}
        ds = xr.open_zarr(path, **opts)
        ds = ds.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items()
                        if k in ds.dims}).sortby("lat").sortby("lon")
        coarsener = None
        if coarsen > 1:
            from ..data.sources import AreaCoarsener
            coarsener = AreaCoarsener(ds.lat.values, ds.lon.values, coarsen)
        if not load:
            return cls(None, hours=[int(h) for h in ds.hour.values], lazy=ds,
                       channels=channels, coarsener=coarsener)
        hours = [int(h) for h in ds.hour.values]
        frames = []
        for c in channels:
            da = ds[c.name] if c.level is None else ds[c.name].sel(level=c.level)
            frames.append(np.asarray(da.transpose("dayofyear", "hour", "lat", "lon").values,
                                     np.float32))
        values = np.stack(frames, axis=2)
        if coarsener is not None:
            values = coarsener(values)
        return cls(values, hours, channels=channels)

    def _day_hour(self, t):
        ts = pd.Timestamp(np.datetime64(int(t), "s"))
        return ts.dayofyear - 1, ts.hour

    def __call__(self, valid_times: np.ndarray) -> torch.Tensor:
        """valid_times [...] (seconds since epoch) -> [..., C, H, W]."""
        flat = np.asarray(valid_times).reshape(-1)
        out = []
        for t in flat:
            d, hour = self._day_hour(t)
            if self.lazy is not None:
                out.append(self._lazy_frame(d, hour))
                continue
            h = self.hours.index(min(self.hours, key=lambda x: abs(x - hour)))
            frame = self.values[d, h]
            if np.isnan(frame).any():  # nearest available day
                avail = np.where(~np.isnan(self.values[:, h, 0, 0, 0]))[0]
                dd = avail[np.argmin(np.minimum(abs(avail - d), 366 - abs(avail - d)))]
                frame = self.values[dd, h]
            out.append(frame)
        arr = np.stack(out).reshape(np.shape(valid_times) + out[0].shape)
        return torch.from_numpy(arr)

    def _lazy_frame(self, day: int, hour: int) -> np.ndarray:
        hour = min(self.hours, key=lambda x: abs(x - hour))
        key = (day, hour)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        sel = self.lazy.sel(dayofyear=day + 1, hour=hour)
        frames = []
        for c in self.channels:
            da = sel[c.name]
            if c.level is not None:
                da = da.sel(level=c.level)
            frames.append(np.asarray(da.transpose("lat", "lon").values, np.float32))
        frame = np.stack(frames)
        if self.coarsener is not None:
            frame = self.coarsener(frame)
        self._cache[key] = frame
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return frame
