import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_wm.data import forcings
from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset, standardize
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5


@pytest.mark.parametrize("name,dyn,tgt", [
    ("graphcast", 227, 227), ("graphcast_13", 83, 83),
    ("graphcast_operational", 82, 83), ("graphcast_13_no_w", 70, 70)])
def test_task_channel_counts(name, dyn, tgt):
    layout = gm.ChannelLayout(gm.TASKS[name])
    assert len(layout.dynamic) == dyn and len(layout.targets) == tgt
    assert len(layout.forcings) == 5 and len(layout.static) == 2
    assert gm.TASKS[name].input_steps == 2


def test_layout_partition_covers_every_target_once():
    layout = gm.ChannelLayout(gm.TASKS["graphcast_13"])
    seen = []
    for part in layout.target_shells:
        seen += part.scalar + part.vector_u + part.vector_v
        for u, v in zip(part.vector_u, part.vector_v):
            assert layout.targets[u].level == layout.targets[v].level
    assert sorted(seen) == list(range(len(layout.targets)))
    assert [layout.targets[i] for i in layout.target_to_dynamic] == layout.dynamic


def test_window_dataset_shapes_and_stats():
    task = gm.get_task("graphcast_13_no_w", (250, 500, 850))
    layout = gm.ChannelLayout(task)
    ds = make_synthetic_era5(num_days=4, num_lat=17, num_lon=32)
    raw = GraphCastWindowDataset(ds, layout, target_steps=3)
    assert len(raw) == 16 - 1 - 3
    stats = compute_stats(raw, range(len(raw)))
    for level in (250, 500, 850):
        u, v = f"u_component_of_wind@{level}", f"v_component_of_wind@{level}"
        assert stats.mean[u] == stats.mean[v] == 0.0 and stats.std[u] == stats.std[v]
    sample = GraphCastWindowDataset(ds, layout, stats, target_steps=3)[0]
    assert sample["inputs"].shape == (2, 20, 17, 32)
    assert sample["targets"].shape == (3, 20, 17, 32)
    assert sample["forcings_in"].shape == (2, 5, 17, 32)
    assert sample["forcings_tgt"].shape == (3, 5, 17, 32)
    assert sample["static"].shape == (2, 17, 32)
    frames, _, times = raw.raw_window(0)
    expected = ds["temperature"].sel(level=500).isel(time=1).values
    np.testing.assert_allclose(frames[1, layout.targets.index(gm.Channel("temperature", 500))],
                               expected)
    assert times[1] == ds.time.values[1]


def test_progress_forcings_match_graphcast_definition():
    t = np.array(["2018-03-01T06:00"], dtype="datetime64[ns]")
    sec = t.astype("datetime64[s]").astype(np.int64)
    np.testing.assert_allclose(forcings.year_progress(t), np.mod(sec / 86400 / 365.24219, 1),
                               rtol=1e-6)
    dp = forcings.day_progress(t, np.array([0.0, 90.0]))
    np.testing.assert_allclose(dp, [[0.25, 0.5]], atol=1e-6)


def test_tisr_is_physical():
    lat, lon = np.linspace(-90, 90, 19), np.arange(36) * 10.0
    tisr = forcings.toa_incident_solar_radiation(["2018-06-21T12:00"], lat, lon)
    assert tisr.min() >= 0
    assert 4.5e6 < tisr.max() < 5.1e6  # ~ 1361 W/m^2 * 3600 s at the subsolar point
    midnight = tisr[0, :, 18]  # 180E at 12 UTC
    assert midnight[:10].max() < 1e3  # dark from the South Pole to the equator
    assert midnight[-1] > 1e6  # polar day at the North Pole in June


def test_standardize_accumulates_hourly_precip():
    times = pd.date_range("2018-01-01", periods=24, freq="1h")
    ds = xr.Dataset({"tp": (("valid_time", "latitude", "longitude"), np.ones((24, 3, 4)))},
                    coords={"valid_time": times, "latitude": [10., 0., -10.],
                            "longitude": [-90., 0., 90., 180.]})
    out = standardize(ds)
    assert list(out.lat.values) == [-10., 0., 10.]
    assert out.lon.values.min() >= 0
    assert list(out.time.dt.hour.values) == [0, 6, 12, 18]
    np.testing.assert_allclose(out.total_precipitation_6hr.isel(time=slice(1, None)).values, 6.0)
