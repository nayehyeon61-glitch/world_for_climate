import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset  # noqa: E402
from climate_wm.data.synthetic import make_synthetic_era5  # noqa: E402
from compare_memory import STEP_S, SpinUpWindows, WarmStart  # noqa: E402
from test_dreamer_game import _mem_model  # noqa: E402
from test_physics_core import _tiny  # noqa: E402


def test_warm_start_pairs_the_windows_and_the_memory_changes_the_free_run():
    layout, stats, data = _tiny()
    ds = make_synthetic_era5(num_days=5, num_lat=9, num_lon=16, levels=(500, 850))
    short = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    long = GraphCastWindowDataset(ds, layout, stats, target_steps=5)
    joined = SpinUpWindows(short, long, 3)
    assert 0 < len(joined) < len(short)  # the first forecast windows have no spin-up
    item = joined[0]
    assert torch.allclose(item["spin_targets"][3:], item["targets"])
    assert int(item["spin_init_time"]) == int(item["init_time"]) - 3 * STEP_S
    assert joined.init_seconds()[0] == int(item["init_time"])

    model = _mem_model(layout, stats, data, {"channels": 4, "hidden": 8})
    torch.manual_seed(1)
    for p in model.memory.parameters():  # an untrained kernel does nothing
        p.data.normal_(0, 0.3)
    batch = {k: v[None] for k, v in item.items()}
    warm = WarmStart(model, 3).eval().rollout(batch, 2, members=2)
    assert warm.shape == (2, 1, 2, len(layout.targets), 9, 16) and torch.isfinite(warm).all()
    assert torch.equal(warm[0], warm[1])
    knocked = WarmStart(model, 3, ("memory",)).rollout(batch, 2)
    assert knocked.shape == (1, 1, 2, len(layout.targets), 9, 16) and torch.isfinite(knocked).all()
    assert not torch.allclose(warm[0], knocked[0])  # the filled memory acts on the free run
    assert model.memory.knock == set()  # restored
    with torch.no_grad():  # the same free run twice: the spin-up is deterministic
        assert torch.allclose(WarmStart(model, 3).rollout(batch, 2)[0], warm[0])


def test_continuous_fill_snapshots_the_memory_at_the_initialisations(tmp_path):
    from compare_memory import ContinuousFill, FilledStart
    layout, stats, data = _tiny()
    ds = make_synthetic_era5(num_days=6, num_lat=9, num_lon=16, levels=(500, 850))
    short = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    inits = [int(short.times[short.indices[k]].astype("datetime64[s]").astype("int64")) for k in (10, 14)]
    model = _mem_model(layout, stats, data, {"channels": 4, "hidden": 8})
    torch.manual_seed(1)
    for p in model.memory.parameters():
        p.data.normal_(0, 0.3)
    start = str(short.times[0].astype("datetime64[s]") + np.timedelta64(6, "h"))
    path = str(tmp_path / "pass.pt")
    fill = ContinuousFill(model, ds, layout, stats, start, inits, chunk=4, path=path)
    assert sorted(fill.snaps) == sorted(inits) and fill.resets == 0
    assert [n for _, n in fill.segments] == [4, 4, 4, 2]  # 14 steps from the start to the last initialisation
    for m in fill.snaps.values():
        assert m.shape[1] == 4 and torch.isfinite(m).all()
    assert not torch.allclose(fill.snaps[inits[0]], fill.snaps[inits[1]])
    again = ContinuousFill(model, ds, layout, stats, start, inits, chunk=4, path=path)  # resumed from the file
    assert torch.equal(again.snaps[inits[1]], fill.snaps[inits[1]])

    batch = {k: v[None] for k, v in short[14].items()}
    filled = FilledStart(model, fill.snaps).eval().rollout(batch, 2, members=2)
    assert filled.shape == (2, 1, 2, len(layout.targets), 9, 16) and torch.isfinite(filled).all()
    empty = FilledStart(model, {inits[1]: torch.zeros_like(fill.snaps[inits[1]])}).rollout(batch, 2)
    assert not torch.allclose(filled[0], empty[0])  # the filled memory acts on the free run


def test_observed_forecast_scores_the_hidden_cap_only():
    from compare_memory import ObservedStart
    layout, stats, data = _tiny()
    ds = make_synthetic_era5(num_days=5, num_lat=9, num_lon=16, levels=(500, 850))
    short = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    long = GraphCastWindowDataset(ds, layout, stats, target_steps=5)
    joined = SpinUpWindows(short, long, 3)
    model = _mem_model(layout, stats, data, {"channels": 4, "hidden": 8})
    torch.manual_seed(1)
    for p in model.memory.parameters():
        p.data.normal_(0, 0.3)
    obs = ObservedStart(model, 3, joined.lat, joined.lon, "region", 0.25, seed=0)
    item = joined[1]
    batch = {k: v[None] for k, v in item.items()}
    masks = obs.masks([int(item["init_time"])])
    assert masks.shape == (1, len(layout.dynamic), 9, 16) and 0 < float(masks.mean()) < 1
    assert torch.equal(masks, obs.masks([int(item["init_time"])]))  # the same cap every time
    # a residual of 1 on every point, hidden-cap scoring: the lat-weighted mean square is 1
    target = torch.zeros(1, 2, len(layout.targets), 9, 16)
    from climate_wm.train.prediction_game import target_mask
    tmask = target_mask(model, masks)
    hidden = obs.hide(target + 1, target, tmask)
    w = obs.lat_w[None, None, None, :, None]
    ms = (hidden.square() * w).sum((-2, -1)) / w.expand_as(hidden).sum((-2, -1))
    assert torch.allclose(ms[:, :, tmask[0].flatten(1).any(1)], torch.ones(1, 1, 1), atol=1e-4)
    assert float(hidden[0, 0, 0][tmask[0, 0] == 0].abs().max()) == 0.0  # nothing outside the cap
    pred = obs.eval().rollout(batch, 2)
    assert pred.shape == (1, 1, 2, len(layout.targets), 9, 16) and torch.isfinite(pred).all()
    assert len(obs.hidden_fraction) == 1 and 0 < obs.hidden_fraction[0] < 1
    nudging = ObservedStart(model, 3, joined.lat, joined.lon, "region", 0.25, seed=0, rule=1.0)
    assert not torch.allclose(nudging.rollout(batch, 2), pred)
