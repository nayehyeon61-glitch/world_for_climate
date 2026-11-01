import torch

from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.train.memory_filter import FilterConfig, consecutive_windows, expand_state_dict, train_filter
from test_dreamer_game import _mem_model
from test_physics_core import _tiny


def test_slow_channels_start_as_the_checkpoint_and_the_filter_trains():
    layout, stats, data = _tiny()
    base = _mem_model(layout, stats, data, {"channels": 4, "hidden": 8})
    torch.manual_seed(1)
    for p in base.memory.parameters():
        p.data.normal_(0, 0.3)
    model = _mem_model(layout, stats, data, {"channels": 4, "hidden": 8, "long_channels": 2,
                                             "long_tau_hours": [720.0, 17520.0]})
    expanded = expand_state_dict(model, base.state_dict())
    assert set(expanded) == {"memory.tend.0.weight", "memory.read.0.weight", "memory.kappa", "memory.log_tau0"}
    assert model.memory.channels == 6 and model.memory.short == 4
    assert torch.allclose(model.memory.log_tau0[:4], base.memory.log_tau0)
    assert abs(float(model.memory.log_tau0[-1].exp()) / (17520.0 * 3600.0) - 1) < 1e-5
    assert torch.equal(model.memory.tend[0].weight[:, -2:], torch.zeros(8, 2))  # the new readers start at zero
    batch = {k: v[None] for k, v in data[0].items()}
    with torch.no_grad():
        assert torch.allclose(model.rollout(batch, 2), base.rollout(batch, 2), atol=1e-3)  # float rounding only
        state, ctx = model.begin(batch)
        assert state["mem"].shape[1] == 6
        snap = batch["inputs"][:, -1]
        mask = torch.zeros_like(snap)
        after = model.posterior(state, ctx, snap, snap, mask)
        assert torch.equal(after["mem"][:, 4:], torch.zeros_like(after["mem"][:, 4:]))  # no innovation into slow channels
        moved, _ = model.advance(after, ctx, 0)
        assert (moved["mem"][:, 4:].abs() > 0).any()  # the slow channels fill from the column state
        model.memory.knock = {"long"}
        pred_knock = model.predict(moved, ctx, 0)
        model.memory.knock = set()
        assert torch.isfinite(pred_knock).all()

    ds = make_synthetic_era5(num_days=4, num_lat=9, num_lon=16, levels=(500, 850))
    cfg = FilterConfig(fill_steps=1, free_steps=2, epochs=1, warmup_windows=1, log_every=1, save_every=2)
    windows, order = consecutive_windows(ds, layout, stats, "2016-01-01T06:00", "2016-01-03T00:00", cfg.steps)
    assert len(order) == 2  # 7 steps available, windows of 3
    saved = []
    history = train_filter(model, windows, order, cfg, save=saved.append, log=print)
    assert len(history["loss"]) == 2 and all(torch.isfinite(torch.tensor(history["loss"])))
    assert history["blowups"] == [0, 0] and saved and saved[-1]["done"] == 2
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not torch.equal(model.memory.tend[0].weight[:, -2:], torch.zeros(8, 2))  # the readers moved
    resumed = train_filter(model, windows, order, FilterConfig(fill_steps=1, free_steps=2, epochs=2, log_every=1),
                           progress=saved[-1], log=print)
    assert len(resumed["loss"]) == 4


def test_persistent_cap_hides_a_region_through_the_window():
    layout, stats, data = _tiny()
    model = _mem_model(layout, stats, data, {"channels": 4, "hidden": 8})
    ds = make_synthetic_era5(num_days=4, num_lat=9, num_lon=16, levels=(500, 850))
    cfg = FilterConfig(fill_steps=2, free_steps=1, epochs=1, warmup_windows=1, log_every=1, save_every=2,
                       persistent_cap=0.25)
    windows, order = consecutive_windows(ds, layout, stats, "2016-01-01T06:00", "2016-01-03T00:00", cfg.steps)
    seen = []
    original = model.posterior

    def spy(state, ctx, forecast, truth, mask):
        seen.append(mask.clone())
        return original(state, ctx, forecast, truth, mask)

    model.posterior = spy
    history = train_filter(model, windows, order, cfg, log=print)
    assert len(history["loss"]) == 2 and history["blowups"] == [0, 0]
    assert seen and all(0 < float(m.mean()) < 0.6 for m in seen)  # a cap is hidden at every re-observation
    first_window = seen[:3]  # snapshot + 2 fill steps share the window's cap
    assert all(torch.equal(m, first_window[0]) for m in first_window)
    assert not torch.equal(seen[0], seen[-1])  # the next window draws another cap
