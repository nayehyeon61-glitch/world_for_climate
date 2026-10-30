"""Lazy ERA5 access with GraphCast semantics.

`open_era5` opens the (Lustre) archive lazily with xarray/dask and returns a
dataset in GraphCast conventions: GraphCast variable names, dims
(time, level, lat, lon), ascending latitude, longitude in [0, 360), 6-hourly
times, `total_precipitation_6hr` accumulated from hourly `tp`, static fields
without a time axis.

`GraphCastWindowDataset` turns that dataset into training / evaluation
samples with exactly the GraphCast split of `extract_inputs_targets_forcings`:

    inputs       [T_in=2, C_dyn, H, W]   X[t-6h], X[t]
    forcings_in  [T_in,   C_f,   H, W]   forcings at the input times
    targets      [L,      C_tgt, H, W]   X[t+6h], ..., X[t+L*6h]
    forcings_tgt [L,      C_f,   H, W]   forcings at the target times
    static       [C_s,    H, W]
"""

from __future__ import annotations

import glob
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import xarray as xr

from . import forcings as forcing_lib
from .graphcast_mapping import (ERA5_SHORT_TO_GRAPHCAST, STATIC_VARS, Channel,
                                ChannelLayout)
from .normalization import NormalizationStats
from .sources import XarraySource

COORD_RENAMES = {
    "latitude": "lat",
    "longitude": "lon",
    "pressure_level": "level",
    "isobaricInhPa": "level",
    "plev": "level",
    "valid_time": "time",
}


def standardize(ds: xr.Dataset, rename: Optional[Dict[str, str]] = None,
                hours: Optional[Sequence[int]] = (0, 6, 12, 18)) -> xr.Dataset:
    """Renames / reorders an ERA5 dataset into GraphCast conventions."""
    coords = {k: v for k, v in COORD_RENAMES.items() if k in ds.variables}
    ds = ds.rename(coords)
    names = {**ERA5_SHORT_TO_GRAPHCAST, **(rename or {})}
    ds = ds.rename({k: v for k, v in names.items() if k in ds.data_vars})
    if "level" in ds.coords and float(ds.level.max()) > 2000:  # Pa -> hPa
        ds = ds.assign_coords(level=(ds.level / 100).round().astype(int))
    ds = ds.sortby("lat")
    ds = ds.assign_coords(lon=ds.lon % 360).sortby("lon")
    if "total_precipitation_6hr" not in ds and "total_precipitation" in ds:
        step = pd.Timedelta(ds.time.values[1] - ds.time.values[0])
        if step == pd.Timedelta("1h"):
            # ERA5 hourly tp at T is the accumulation over (T-1h, T]. The first
            # 5 hours have no complete 6h window; drop them so no NaN frame
            # reaches the windows or the normalisation statistics.
            ds["total_precipitation_6hr"] = ds.total_precipitation.rolling(
                time=6, min_periods=6).sum()
            ds = ds.isel(time=slice(5, None))
        elif step == pd.Timedelta("6h"):
            ds["total_precipitation_6hr"] = ds.total_precipitation
        else:
            raise ValueError(f"Cannot derive 6h precipitation from step {step}.")
    for var in STATIC_VARS:
        if var in ds and "time" in ds[var].dims:
            ds[var] = ds[var].isel(time=0, drop=True)
    if hours is not None:
        ds = ds.sel(time=ds.time.dt.hour.isin(list(hours)))
    return ds


def open_era5(cfg: dict) -> xr.Dataset:
    """Opens the archive described by `configs/era5_source.yaml`.

    `paths` entries are glob patterns / zarr stores, or {path, rename} dicts
    to rename per source before merging (e.g. the surface geopotential `z`
    of the invariants file, which would clash with pressure-level `z`).
    """
    chunks = cfg.get("chunks", {"time": 1})
    parts = []
    for entry in cfg["paths"]:
        pattern, rename = (entry, {}) if isinstance(entry, str) else (
            entry["path"], entry.get("rename", {}))
        if pattern.rstrip("/").endswith(".zarr"):
            part = xr.open_zarr(pattern, chunks=chunks)
        else:
            files = sorted(glob.glob(pattern))
            if not files:
                raise FileNotFoundError(pattern)
            part = xr.open_mfdataset(files, chunks=chunks, combine="by_coords",
                                     engine=cfg.get("engine"))
        parts.append(part.rename({k: v for k, v in rename.items() if k in part}))
    ds = xr.merge(parts, compat="override", join="outer")
    return standardize(ds, cfg.get("rename"), cfg.get("hours", (0, 6, 12, 18)))


class GraphCastWindowDataset(torch.utils.data.Dataset):
    """(inputs, forcings, targets) windows from a frame source.

    `ds` is a standardized xarray dataset (wrapped in `XarraySource`) or any
    source from `climate_wm.data.sources`, e.g. `HourlyArchive` for the hourly
    0.25 degree archive. Windows with a missing frame are skipped.
    """

    def __init__(self, ds, layout: ChannelLayout,
                 stats: Optional[NormalizationStats] = None,
                 target_steps: int = 1, start: Optional[str] = None,
                 end: Optional[str] = None,
                 init_hours: Optional[Sequence[int]] = None,
                 compute_tisr: Optional[bool] = None):
        self.layout = layout
        self.stats = stats
        self.target_steps = target_steps
        self.input_steps = layout.task.input_steps
        levels = [int(l) for l in layout.task.pressure_levels]
        self.source = XarraySource(ds, levels) if isinstance(ds, xr.Dataset) else ds
        self.times = self.source.times
        step = np.timedelta64(pd.Timedelta(layout.task.timestep))
        if len(self.times) > 1 and not np.all(np.diff(self.times) == step):
            raise ValueError("Dataset times must be uniformly spaced by the "
                             f"task timestep {layout.task.timestep}.")
        self.lat = np.asarray(self.source.lat, np.float32)
        self.lon = np.asarray(self.source.lon, np.float32)
        tisr = "toa_incident_solar_radiation"
        self.compute_tisr = (not self.source.has(tisr)) if compute_tisr is None else compute_tisr
        names = {c.name for c in layout.targets}
        if not self.compute_tisr and tisr in {c.name for c in layout.forcings}:
            names.add(tisr)
        available = self.source.availability(names)
        lo = np.datetime64(start) if start else self.times[0]
        hi = np.datetime64(end) if end else self.times[-1]
        valid = []
        for i in range(self.input_steps - 1, len(self.times) - target_steps):
            t = self.times[i]
            if not lo <= t <= hi:
                continue
            if init_hours is not None and pd.Timestamp(t).hour not in init_hours:
                continue
            if not available[i - self.input_steps + 1:i + target_steps + 1].all():
                continue
            valid.append(i)
        self.indices = np.asarray(valid, np.int64)
        self.static = self.source.read_static(layout.static)
        if hasattr(self.source, "close"):
            self.source.close()  # no open file handles when DataLoader workers fork

    @property
    def ds(self):
        return getattr(self.source, "ds", None)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def grid_shape(self):
        return len(self.lat), len(self.lon)

    def _load(self, channels: Sequence[Channel], time_idx) -> np.ndarray:
        if time_idx is None:
            return self.source.read_static(channels)
        return self.source.read(channels, time_idx)

    def forcing_frames(self, times) -> np.ndarray:
        """[T, C_f, H, W] in the layout's forcing order."""
        fields = forcing_lib.time_forcings(times, self.lon, len(self.lat))
        tisr = "toa_incident_solar_radiation"
        if tisr in [c.name for c in self.layout.forcings]:
            if self.compute_tisr:
                fields[tisr] = forcing_lib.toa_incident_solar_radiation(
                    times, self.lat, self.lon)
            else:
                idx = np.searchsorted(self.times, np.asarray(times, "datetime64[ns]"))
                fields[tisr] = self.source.read([Channel(tisr)], idx)[:, 0]
        return np.stack([fields[c.name] for c in self.layout.forcings], axis=1)

    def raw_window(self, k: int):
        """Unnormalised (frames [T_in+L, C_tgt, H, W], forcings, times)."""
        i = int(self.indices[k])
        idx = np.arange(i - self.input_steps + 1, i + self.target_steps + 1)
        frames = self._load(self.layout.targets, idx)
        times = self.times[idx]
        return frames, self.forcing_frames(times), times

    def __getitem__(self, k: int) -> Dict[str, torch.Tensor]:
        frames, forc, times = self.raw_window(k)
        n = self.input_steps
        inputs = frames[:n][:, self.layout.target_to_dynamic]
        targets = frames[n:]
        static = self.static
        if self.stats is not None:
            inputs = self.stats.normalize(inputs, self.layout.dynamic)
            targets = self.stats.normalize(targets, self.layout.targets)
            forc = self.stats.normalize(forc, self.layout.forcings)
            static = self.stats.normalize(static, self.layout.static)
        as_t = lambda a: torch.from_numpy(np.ascontiguousarray(a, np.float32))
        return {
            "inputs": as_t(inputs),
            "forcings_in": as_t(forc[:n]),
            "targets": as_t(targets),
            "forcings_tgt": as_t(forc[n:]),
            "static": as_t(static),
            "init_time": torch.tensor(times[n - 1].astype("datetime64[s]").astype(np.int64)),
        }
