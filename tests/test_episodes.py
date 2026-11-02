import datetime as dt

import numpy as np

from climate_wm.sim.episodes import bitround, plan_episodes
from climate_wm.sim.speedy import TEND_GROUPS, tendencies_to_wb2


def test_plan_splits_by_episode_and_varies_each_group():
    dates = [dt.datetime(1982, 1, 1) + dt.timedelta(days=5 * i) for i in range(219)]
    specs = plan_episodes(dates, {"A": 12, "B": 12, "C": 12, "D": 1}, test_per_group={"A": 3, "B": 3, "C": 3, "D": 1})
    by = lambda g: [s for s in specs if s.group == g]
    assert [len(by(g)) for g in "ABCD"] == [12, 12, 12, 1]
    assert all(sum(s.split == "test" for s in by(g)) == n for g, n in zip("ABCD", (3, 3, 3, 1)))
    assert len({s.start for s in by("A")}) == 12 and not {s.start for s in by("A")} & {s.start for s in by("B")}
    assert all(s.perturbation and s.perturbation.seed and s.spinup_days == 1 for s in by("B"))
    assert all(0.5 <= s.intervention.conv_factor <= 2 and s.perturbation is None for s in by("C"))
    assert by("D")[0].days == 180 and all(s.days == 30 for s in by("A"))
    assert all(s.start + dt.timedelta(days=30) <= dates[-1] + dt.timedelta(days=5) for s in by("C"))


def test_bitround_bounds_the_relative_error():
    x = np.random.default_rng(0).normal(0, 1e4, 1000).astype(np.float32)
    y = bitround(x, 12)
    assert np.all(np.abs(y - x) <= np.abs(x) * 2.0 ** -12)
    assert np.all((y.view(np.uint32) & np.uint32((1 << 11) - 1)) == 0)


def test_tendencies_map_to_pressure_levels_and_vanish_below_ground():
    sigma = np.array([0.025, 0.095, 0.2, 0.34, 0.51, 0.685, 0.835, 0.95])
    lat = np.linspace(-87, 87, 48)
    ps = np.full((2, 48, 96), 1.0e5)
    ps[:, :, :10] = 7.0e4                      # a high plateau
    tend = {k: np.ones((2, 8, 48, 96)) for s in TEND_GROUPS.values() for k in s}
    out = tendencies_to_wb2(tend, {"lat": lat, "sigma": sigma, "ps": ps}, levels=(500, 850, 1000))
    assert out["Q"].shape == (2, 3, 64, 32) and np.isclose(out["Q_cnv"][0, 0, 30, 16], 1.0)
    assert np.isclose(out["S_cnv"][0, 0, 30, 16], 1e-3)          # g/kg/s -> kg/kg/s
    assert out["Q_cnv"][0, 1, 0, 16] == 0.0 and out["Q_cnv"][0, 0, 0, 16] == 1.0  # 850 hPa under the plateau
    assert np.isclose(out["Q"][0, 0, 30, 16], 5.0)
