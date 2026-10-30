"""Frame sources behind `GraphCastWindowDataset`.

A source exposes a uniform time axis at the task timestep and reads frames of
GraphCast channels as float32 arrays (ascending latitude, longitude in
[0, 360)):

    times                    [T] datetime64[ns], uniformly spaced
    lat [H], lon [W]
    availability(names)      [T] bool, frames whose variables can all be read
    read(channels, idx)      [len(idx), C, H, W]
    read_static(channels)    [C, H, W]
    has(name)

`XarraySource` wraps an in-memory or dask-backed dataset (synthetic data,
WeatherBench-2 zarr subsets, small archives opened with `open_era5`).

`HourlyArchive` is for the ~80 TB hourly 0.25 degree ERA5 archive on Lustre:
opening it with `xr.open_mfdataset` would build a dask graph over hundreds of
thousands of chunks and pay for it on every sample. Instead each file is
scanned once (cached and extended incrementally as new years arrive), and a
frame is read with a plain `isel` on the one file that holds it.
"""

from __future__ import annotations

import glob
import os
import pickle
from collections import OrderedDict
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .graphcast_mapping import ERA5_SHORT_TO_GRAPHCAST, STATIC_VARS, Channel

TP6 = "total_precipitation_6hr"
TP1 = "total_precipitation"
_TIME_NAMES = ("time", "valid_time")
_LEVEL_NAMES = ("level", "pressure_level", "isobaricInhPa", "plev")
_LAT_NAMES = ("lat", "latitude")
_LON_NAMES = ("lon", "longitude")


def _group_by_variable(channels: Sequence[Channel]) -> "OrderedDict[str, List[Optional[int]]]":
    groups: "OrderedDict[str, List[Optional[int]]]" = OrderedDict()
    for c in channels:
        groups.setdefault(c.name, []).append(c.level)
    return groups


def _first(names, container):
    return next((n for n in names if n in container), None)


# ---------------------------------------------------------------- coarsening
class AreaCoarsener:
    """Conservative (cell-area weighted) regridding from a regular lat/lon grid
    to the grid made of every `factor`-th point, e.g. 0.25 deg (721 x 1440,
    with poles) -> 1.0 deg (181 x 360) for factor 4.

    Target cells are `factor` source spacings wide and centred on the kept
    points; the weight of a source cell is its exact overlap area (latitude
    bands clipped at the poles, longitude wrapped). Only the few overlapping
    rows / columns are touched, so a 0.25 deg frame costs a handful of strided
    multiply-adds instead of a dense matrix product.
    """

    def __init__(self, lat: np.ndarray, lon: np.ndarray, factor: int):
        self.factor = int(factor)
        f = self.factor
        self.lat_src, self.lon_src = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        self.lat = self.lat_src[::f]
        self.lon = self.lon_src[::f]
        dlat = abs(self.lat_src[1] - self.lat_src[0])
        dlon = abs(self.lon_src[1] - self.lon_src[0])
        taps = np.arange(-f, f + 1)
        self.lat_rows, self.lat_w = self._weights(
            self.lat_src, self.lat, taps, dlat, f, latitude=True)
        self.lon_cols, self.lon_w = self._weights(
            self.lon_src, self.lon, taps, dlon, f, latitude=False)

    @staticmethod
    def _weights(src, dst, taps, d, f, latitude):
        step = 1  # dst points are src[::f]
        idx0 = np.arange(len(dst)) * f * step
        rows = idx0[:, None] + taps[None, :]
        n = len(src)
        if latitude:
            valid = (rows >= 0) & (rows < n)
            rows = np.clip(rows, 0, n - 1)
            lo_t = np.clip(dst - f * d / 2, -90, 90)[:, None]
            hi_t = np.clip(dst + f * d / 2, -90, 90)[:, None]
            lo_s = np.clip(src[rows] - d / 2, -90, 90)
            hi_s = np.clip(src[rows] + d / 2, -90, 90)
            lo, hi = np.maximum(lo_t, lo_s), np.minimum(hi_t, hi_s)
            w = np.where(hi > lo, np.sin(np.radians(hi)) - np.sin(np.radians(lo)), 0.0)
            w = np.where(valid, w, 0.0)
        else:
            rows = rows % n  # periodic in longitude
            offset = taps[None, :] * d  # source cell centre relative to target centre
            lo = np.maximum(-f * d / 2, offset - d / 2)
            hi = np.minimum(f * d / 2, offset + d / 2)
            w = np.clip(hi - lo, 0.0, None) * np.ones((len(dst), 1))
        w = w / w.sum(1, keepdims=True)
        keep = np.any(w > 0, axis=0)
        return rows[:, keep], w[:, keep].astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x [..., H_src, W_src] -> [..., H_dst, W_dst]."""
        if self.factor == 1:
            return x
        out = 0.0
        for k in range(self.lon_cols.shape[1]):
            out = out + x[..., self.lon_cols[:, k]] * self.lon_w[:, k]
        res = 0.0
        for k in range(self.lat_rows.shape[1]):
            res = res + out[..., self.lat_rows[:, k], :] * self.lat_w[:, k, None]
        return np.asarray(res, np.float32)


# ---------------------------------------------------------------- xarray
class XarraySource:
    """A standardized dataset (see `era5_lazy_loader.standardize`)."""

    def __init__(self, ds: xr.Dataset, levels: Optional[Sequence[int]] = None):
        if levels is not None and "level" in ds.dims and \
                [int(l) for l in ds.level.values] != [int(l) for l in levels]:
            ds = ds.sel(level=list(levels))  # copies data: pre-select when loading into memory
        self.ds = ds
        self.times = ds.time.values.astype("datetime64[ns]")
        self.lat = ds.lat.values.astype(np.float32)
        self.lon = ds.lon.values.astype(np.float32)

    def has(self, name: str) -> bool:
        return name in self.ds

    def availability(self, names: Iterable[str]) -> np.ndarray:
        return np.ones(len(self.times), bool)

    def _array(self, sub, name, levels, timed):
        da = sub[name]
        dims = (["time"] if timed else [])
        if levels[0] is not None:
            return np.asarray(da.sel(level=levels).transpose(*dims, "level", "lat", "lon").values,
                              np.float32)
        return np.asarray(da.transpose(*dims, "lat", "lon").values, np.float32)[..., None, :, :]

    def read(self, channels: Sequence[Channel], time_idx) -> np.ndarray:
        sub = self.ds.isel(time=np.asarray(time_idx))
        return np.concatenate([self._array(sub, n, lv, True)
                               for n, lv in _group_by_variable(channels).items()], axis=-3)

    def read_static(self, channels: Sequence[Channel]) -> np.ndarray:
        if not channels:
            return np.zeros((0, len(self.lat), len(self.lon)), np.float32)
        return np.concatenate([self._array(self.ds, n, lv, False)
                               for n, lv in _group_by_variable(channels).items()], axis=-3)


# ---------------------------------------------------------------- hourly archive
def _scan_file(path: str, engine: Optional[str], rename: Dict[str, str]) -> dict:
    """Variables, times, levels and grid of one file (values are not read)."""
    names = {**ERA5_SHORT_TO_GRAPHCAST, **rename}
    with xr.open_dataset(path, engine=engine, cache=False) as ds:
        tname = _first(_TIME_NAMES, ds.coords) or _first(_TIME_NAMES, ds.dims)
        lname = _first(_LEVEL_NAMES, ds.coords)
        latn, lonn = _first(_LAT_NAMES, ds.coords), _first(_LON_NAMES, ds.coords)
        times = (ds[tname].values.astype("datetime64[ns]").astype(np.int64)
                 if tname is not None else np.zeros(0, np.int64))
        levels = None
        if lname is not None:
            levels = np.asarray(ds[lname].values, np.float64)
            if levels.max() > 2000:  # Pa -> hPa
                levels = levels / 100.0
            levels = np.round(levels).astype(int)
        variables = {}
        for v in ds.data_vars:
            if latn not in ds[v].dims:
                continue
            variables[names.get(v, v)] = {"name": v, "dims": tuple(ds[v].dims)}
        return {"times": np.atleast_1d(times), "time_dim": tname, "level_dim": lname,
                "levels": levels, "vars": variables, "lat_dim": latn, "lon_dim": lonn,
                "lat": np.asarray(ds[latn].values, np.float64),
                "lon": np.asarray(ds[lonn].values, np.float64),
                "mtime": os.path.getmtime(path), "size": os.path.getsize(path)}


class HourlyArchive:
    """Indexed reader for a large hourly ERA5 archive (netCDF / GRIB / zarr files).

    cfg keys (see configs/era5_lustre_hourly.yaml):
      paths        glob patterns, or {path, rename} for per-file renames
                   (e.g. the invariants file's `z` -> geopotential_at_surface)
      engine       xarray engine (netcdf4, h5netcdf, cfgrib, zarr)
      index_cache  pickle with the per-file scan, reused and extended
      hours        frame hours (default 0, 6, 12, 18)
      time_range   [start, end] of the frame axis
      coarsen      1 (native) or k: conservative regridding to every k-th point
      scan_workers threads for the one-off scan
      open_files   per-process cache size of open file handles
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.engine = cfg.get("engine")
        entries = []
        for entry in cfg["paths"]:
            pattern, rename = (entry, {}) if isinstance(entry, str) else (
                entry["path"], entry.get("rename", {}))
            files = sorted(glob.glob(pattern, recursive=True))
            if not files and not pattern.rstrip("/").endswith(".zarr"):
                raise FileNotFoundError(f"no files match {pattern}")
            entries += [(f, rename) for f in (files or [pattern])]
        self.files = [f for f, _ in entries]
        self.renames = [r for _, r in entries]
        self.scan = self._load_or_scan(entries)
        self._build_index()
        self._handles: "OrderedDict[int, xr.Dataset]" = OrderedDict()
        self._pid = os.getpid()

    # ---- scanning
    def _load_or_scan(self, entries) -> List[dict]:
        cache_path = self.cfg.get("index_cache")
        cached = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
        todo = [(i, f, r) for i, (f, r) in enumerate(entries)
                if f not in cached or cached[f]["mtime"] != os.path.getmtime(f)
                or cached[f].get("rename") != r]
        if todo:
            print(f"[archive] scanning {len(todo)} of {len(entries)} files ...")
            args = ([f for _, f, _ in todo], [self.engine] * len(todo), [r for _, _, r in todo])
            workers = min(self.cfg.get("scan_workers", 16), len(todo))
            if workers > 1:  # processes: the netCDF / HDF5 libraries are not thread-safe
                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(workers, mp_context=ctx) as pool:
                    results = list(pool.map(_scan_file, *args, chunksize=8))
            else:
                results = [_scan_file(*a) for a in zip(*args)]
            for (_, f, r), res in zip(todo, results):
                res["rename"] = r
                cached[f] = res
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                tmp = cache_path + ".tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(cached, fh, protocol=4)
                os.replace(tmp, cache_path)
        return [cached[f] for f, _ in entries]

    def _build_index(self) -> None:
        per_var: Dict[str, List[np.ndarray]] = {}
        self.static_vars: Dict[str, int] = {}
        grid = None
        for fid, s in enumerate(self.scan):
            for var, meta in s["vars"].items():
                timed = s["time_dim"] is not None and s["time_dim"] in meta["dims"]
                if var in STATIC_VARS or not timed:
                    self.static_vars.setdefault(var, fid)
                    continue
                t = s["times"]
                per_var.setdefault(var, []).append(np.stack(
                    [t, np.full(len(t), fid), np.arange(len(t))], 1))
            if grid is None and len(s["lat"]) > 1 and s["vars"]:
                grid = (s["lat"], s["lon"])
        self.index = {}
        for var, chunks in per_var.items():
            table = np.concatenate(chunks)
            table = table[np.argsort(table[:, 0], kind="stable")]
            first = np.concatenate([[True], np.diff(table[:, 0]) > 0])  # drop duplicates
            self.index[var] = table[first]
        lat, lon = grid
        self.lat_order = np.argsort(lat)
        lon360 = np.mod(lon, 360.0)
        self.lon_order = np.argsort(lon360)
        lat_sorted, lon_sorted = lat[self.lat_order], lon360[self.lon_order]
        factor = int(self.cfg.get("coarsen", 1))
        self.coarsener = AreaCoarsener(lat_sorted, lon_sorted, factor) if factor > 1 else None
        self.lat = (self.coarsener.lat if self.coarsener else lat_sorted).astype(np.float32)
        self.lon = (self.coarsener.lon if self.coarsener else lon_sorted).astype(np.float32)
        hours = list(self.cfg.get("hours", (0, 6, 12, 18)))
        step = pd.Timedelta(hours=int(np.diff(hours).min()) if len(hours) > 1 else 24)
        any_var = next(iter(self.index.values()))[:, 0]
        lo, hi = (self.cfg.get("time_range") or [None, None])
        start = pd.Timestamp(lo) if lo else pd.Timestamp(any_var.min())
        end = pd.Timestamp(hi) if hi else pd.Timestamp(any_var.max())
        start = start.ceil(step)
        self.times = pd.date_range(start, end, freq=step).values.astype("datetime64[ns]")
        self.times = self.times[pd.DatetimeIndex(self.times).hour.isin(hours)]

    # ---- queries
    def has(self, name: str) -> bool:
        if name == TP6:
            return TP6 in self.index or TP1 in self.index
        return name in self.index or name in self.static_vars

    def _lookup(self, var: str, times_ns: np.ndarray):
        table = self.index[var]
        pos = np.searchsorted(table[:, 0], times_ns)
        pos_c = np.clip(pos, 0, len(table) - 1)
        found = table[pos_c, 0] == times_ns
        return found, table[pos_c, 1], table[pos_c, 2]

    def _needed(self, name: str):
        """(variable in the archive, hour offsets) behind a GraphCast variable."""
        if name == TP6 and TP6 not in self.index:
            return TP1, np.arange(-5, 1)
        return name, np.zeros(1, int)

    def availability(self, names: Iterable[str]) -> np.ndarray:
        ok = np.ones(len(self.times), bool)
        t = self.times.astype(np.int64)
        for name in set(names):
            if name in self.static_vars:
                continue
            var, offsets = self._needed(name)
            if var not in self.index:
                raise KeyError(f"{name} is not in the archive (has {sorted(self.index)})")
            for off in offsets:
                found, _, _ = self._lookup(var, t + int(off) * 3_600_000_000_000)
                ok &= found
        return ok

    # ---- reading
    def close(self) -> None:
        """Closes cached handles (call before forking DataLoader workers)."""
        for ds in self._handles.values():
            ds.close()
        self._handles = OrderedDict()

    def _open(self, fid: int) -> xr.Dataset:
        if os.getpid() != self._pid:  # forked DataLoader worker: never reuse parent handles
            self._handles, self._pid = OrderedDict(), os.getpid()
        ds = self._handles.pop(fid, None)
        if ds is None:
            ds = xr.open_dataset(self.files[fid], engine=self.engine, cache=False)
            if len(self._handles) >= self.cfg.get("open_files", 64):
                _, old = self._handles.popitem(last=False)
                old.close()
        self._handles[fid] = ds
        return ds

    def _orient(self, a: np.ndarray) -> np.ndarray:
        a = a[..., self.lat_order, :][..., self.lon_order]
        return self.coarsener(a) if self.coarsener is not None else a

    def _read_var(self, fid: int, local: Optional[int], var: str,
                  levels: Sequence[Optional[int]]) -> np.ndarray:
        """[L or 1, H, W] for one variable at one time (local index) in one file."""
        s = self.scan[fid]
        meta = s["vars"][var]
        da = self._open(fid)[meta["name"]]
        if local is not None:
            da = da.isel({s["time_dim"]: int(local)})
        elif s["time_dim"] in da.dims:
            da = da.isel({s["time_dim"]: 0})
        if "expver" in da.dims:  # ERA5T + ERA5 in one file: first valid value
            vals = np.asarray(da.transpose("expver", ...).values, np.float32)
            a = vals[0]
            for alt in vals[1:]:
                a = np.where(np.isnan(a), alt, a)
            da = xr.DataArray(a, dims=[d for d in da.dims if d != "expver"])
        if levels[0] is not None:
            pos = [int(np.where(s["levels"] == l)[0][0]) for l in levels]
            da = da.isel({s["level_dim"]: pos}).transpose(s["level_dim"], s["lat_dim"], s["lon_dim"])
            a = np.asarray(da.values, np.float32)
        else:
            a = np.asarray(da.transpose(s["lat_dim"], s["lon_dim"]).values, np.float32)[None]
        return a

    def read(self, channels: Sequence[Channel], time_idx) -> np.ndarray:
        t = self.times[np.asarray(time_idx)].astype(np.int64)
        frames = []
        for tk in t:
            parts = []
            for name, levels in _group_by_variable(channels).items():
                var, offsets = self._needed(name)
                acc = None
                for off in offsets:
                    found, fid, local = self._lookup(var, np.array([tk + int(off) * 3_600_000_000_000]))
                    if not found[0]:
                        raise KeyError(f"{var} missing at {np.datetime64(int(tk), 'ns')}")
                    a = self._read_var(int(fid[0]), int(local[0]), var, levels)
                    acc = a if acc is None else acc + a
                parts.append(acc)
            frames.append(self._orient(np.concatenate(parts, axis=0)))
        return np.stack(frames).astype(np.float32)

    def read_static(self, channels: Sequence[Channel]) -> np.ndarray:
        parts = []
        for name, levels in _group_by_variable(channels).items():
            if name not in self.static_vars:
                raise KeyError(
                    f"static field {name} not found: add the ERA5 invariants file to `paths` "
                    "(e.g. {path: .../invariants.nc, rename: {z: geopotential_at_surface}})")
            parts.append(self._read_var(self.static_vars[name], None, name, levels))
        if not parts:
            return np.zeros((0, len(self.lat), len(self.lon)), np.float32)
        return self._orient(np.concatenate(parts, axis=0))

    def summary(self) -> str:
        lines = [f"hourly archive: {len(self.files)} files, frames {self.times[0]} .. "
                 f"{self.times[-1]} ({len(self.times)}), grid {len(self.lat)} x {len(self.lon)}"
                 + (f" (coarsened x{self.coarsener.factor})" if self.coarsener else "")]
        for var, table in sorted(self.index.items()):
            t = table[:, 0].astype("datetime64[ns]")
            lines.append(f"  {var:<40} {len(table):>8} hourly steps  {t.min()} .. {t.max()}")
        for var in sorted(self.static_vars):
            lines.append(f"  {var:<40} static")
        return "\n".join(lines)
