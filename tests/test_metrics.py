import itertools

import numpy as np
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.eval import graphcast_metrics as gcm
from climate_wm.eval.ensemble_metrics import crps_fair
from climate_wm.train.losses import graphcast_channel_weights, normalized_latitude_weights


def test_latitude_weights_with_and_without_poles():
    with_poles = normalized_latitude_weights(np.linspace(-90, 90, 721))
    np.testing.assert_allclose(with_poles.mean(), 1.0)
    # weathernext: pole cell area ~ 1/8 of the neighbouring cell.
    np.testing.assert_allclose(with_poles[0] / with_poles[1], 1 / 8, rtol=1e-3)
    no_poles = normalized_latitude_weights(np.arange(-89.5, 90, 1.0))
    np.testing.assert_allclose(no_poles.mean(), 1.0)
    np.testing.assert_allclose(no_poles, no_poles[::-1])


def test_rmse_conventions_and_acc():
    lat = np.linspace(-90, 90, 9)
    ch = [gm.Channel("2m_temperature")]
    m = gcm.GraphCastMetrics(ch, lat, [6])
    target = torch.zeros(2, 1, 1, 9, 12)
    pred = torch.stack([torch.full((1, 1, 9, 12), 1.0), torch.full((1, 1, 9, 12), 3.0)])
    m.update(pred, target, clim=None)
    res = m.compute()
    np.testing.assert_allclose(res["rmse"][0, 0], 2.0, rtol=1e-6)  # eq. 20: mean of sqrt
    np.testing.assert_allclose(res["rmse_trad"][0, 0], np.sqrt(5.0), rtol=1e-6)  # eq. 21
    clim = torch.zeros(1, 1, 1, 9, 12)
    x = torch.randn(1, 1, 1, 9, 12)
    lat_w = torch.tensor(normalized_latitude_weights(lat), dtype=torch.float32)
    torch.testing.assert_close(gcm.acc_per_init(x, x, clim, lat_w), torch.ones(1, 1, 1))
    torch.testing.assert_close(gcm.acc_per_init(-x, x, clim, lat_w), -torch.ones(1, 1, 1))
    assert np.isclose(gcm.rmse_skill_score(0.9, 1.0), -0.1)
    assert np.isclose(gcm.acc_skill_score(0.95, 0.9), 0.5)


def test_graphcast_channel_weights():
    layout = gm.ChannelLayout(gm.TASKS["graphcast_13"])
    w = graphcast_channel_weights(layout)
    keys = [c.key for c in layout.targets]
    assert np.isclose(w[keys.index("2m_temperature")], 1.0)
    assert np.isclose(w[keys.index("mean_sea_level_pressure")], 0.1)
    t = [w[keys.index(f"temperature@{l}")] for l in layout.task.pressure_levels]
    assert np.isclose(sum(t), 1.0)  # per-variable mean over levels of level weights
    assert t[-1] > t[0]  # 1000 hPa weighted more than 50 hPa


def test_crps_fair_matches_pairwise_definition():
    torch.manual_seed(0)
    ens, y = torch.randn(5, 7), torch.randn(7)
    m = ens.shape[0]
    pair = sum((ens[i] - ens[j]).abs() for i, j in itertools.permutations(range(m), 2))
    ref = (ens - y).abs().mean(0) - 0.5 * pair / (m * (m - 1))
    torch.testing.assert_close(crps_fair(ens, y), ref)
