import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data import sources
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.graphcast_mapping import Channel
from climate_wm.data.sources import AreaCoarsener, HourlyArchive

LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]


def _write_archive(root, days=2, res=5.0):
    """CDS-style daily files: descending latitude, `valid_time`, `pressure_level`."""
    lat = np.linspace(90, -90, int(180 / res) + 1)
    lon = np.arange(0, 360, res)
    rng = np.random.default_rng(0)
    for d in range(days):
        times = pd.date_range("2018-01-01", periods=24, freq="1h") + pd.Timedelta(days=d)
        shape3 = (24, len(LEVELS), len(lat), len(lon))
        shape2 = (24, len(lat), len(lon))
        pl = xr.Dataset({v: (("valid_time", "pressure_level", "latitude", "longitude"),
                             rng.standard_normal(shape3).astype(np.float32))
                         for v in ("z", "q", "t", "u", "v")},
                        coords={"valid_time": times, "pressure_level": LEVELS,
                                "latitude": lat, "longitude": lon})
        sl = xr.Dataset({v: (("valid_time", "latitude", "longitude"),
                             rng.standard_normal(shape2).astype(np.float32))
                         for v in ("u10", "v10", "msl", "t2m", "tisr")},
                        coords={"valid_time": times, "latitude": lat, "longitude": lon})
        sl["tp"] = (("valid_time", "latitude", "longitude"), np.full(shape2, 1e-3, np.float32))
        pl.to_netcdf(root / f"pl_{d}.nc")
        sl.to_netcdf(root / f"sl_{d}.nc")
    inv = xr.Dataset({"z": (("valid_time", "latitude", "longitude"), rng.standard_normal((1, len(lat), len(lon)))),
                      "lsm": (("valid_time", "latitude", "longitude"), rng.random((1, len(lat), len(lon))))},
                     coords={"valid_time": pd.DatetimeIndex(["2018-01-01"]), "latitude": lat, "longitude": lon})
    inv.to_netcdf(root / "invariants.nc")
    return lat, lon


def _cfg(root, **kw):
    return {"paths": [str(root / "pl_*.nc"), str(root / "sl_*.nc"),
                      {"path": str(root / "invariants.nc"), "rename": {"z": "geopotential_at_surface"}}],
            "engine": "netcdf4", "index_cache": str(root / "index.pkl"), **kw}


def test_archive_reads_frames_and_accumulates_precip(tmp_path, monkeypatch):
    lat, _ = _write_archive(tmp_path)
    archive = HourlyArchive(_cfg(tmp_path))
    assert len(archive.times) == 8 and np.all(np.diff(archive.lat) > 0)
    ok = archive.availability({"geopotential", "total_precipitation_6hr"})
    assert list(ok) == [False] + [True] * 7  # 00 UTC of day 1 lacks the previous 5 hours of tp
    frame = archive.read([Channel("geopotential", 500), Channel("total_precipitation_6hr")], [1])
    src = xr.open_dataset(tmp_path / "pl_0.nc").z.sel(pressure_level=500).isel(valid_time=6).values
    np.testing.assert_allclose(frame[0, 0], src[::-1], rtol=1e-6)  # latitude flipped to ascending
    np.testing.assert_allclose(frame[0, 1], 6e-3, rtol=1e-5)
    static = archive.read_static([Channel("geopotential_at_surface"), Channel("land_sea_mask")])
    assert static.shape == (2, len(lat), 72)

    # the index cache is reused: no file is scanned again
    monkeypatch.setattr(sources, "_scan_file", lambda *a: pytest.fail("rescanned"))
    again = HourlyArchive(_cfg(tmp_path))
    assert len(again.times) == 8


def test_window_dataset_on_coarsened_archive(tmp_path):
    _write_archive(tmp_path)
    archive = HourlyArchive(_cfg(tmp_path, coarsen=2))
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w"))
    ds = GraphCastWindowDataset(archive, layout, None, target_steps=1)
    assert ds.grid_shape == (19, 36)
    assert not ds.compute_tisr  # tisr comes from the archive
    assert len(ds) == 5  # windows needing the incomplete first frame are skipped
    frames, forc, times = ds.raw_window(0)
    assert frames.shape == (3, len(layout.targets), 19, 36)
    assert forc.shape == (3, len(layout.forcings), 19, 36)
    assert np.isfinite(frames).all() and ds.static.shape == (2, 19, 36)


def _cell_area(lat, d):
    lo, hi = np.clip(lat - d / 2, -90, 90), np.clip(lat + d / 2, -90, 90)
    return np.sin(np.radians(hi)) - np.sin(np.radians(lo))


@pytest.mark.parametrize("factor", [2, 4])
def test_coarsening_conserves_the_area_mean(factor):
    d = 2.5
    lat, lon = np.linspace(-90, 90, 73), np.arange(0, 360, d)
    x = np.random.default_rng(1).standard_normal((3, len(lat), len(lon))).astype(np.float32)
    co = AreaCoarsener(lat, lon, factor)
    y = co(x)
    assert y.shape == (3, len(co.lat), len(co.lon))
    w_src = _cell_area(lat, d)[:, None]
    w_dst = _cell_area(co.lat, factor * d)[:, None]
    np.testing.assert_allclose((y * w_dst).sum((-2, -1)) / (w_dst.sum() * len(co.lon)),
                               (x * w_src).sum((-2, -1)) / (w_src.sum() * len(lon)), atol=1e-5)
    np.testing.assert_allclose(co(np.ones_like(x)), 1.0, rtol=1e-6)
