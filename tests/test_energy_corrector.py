import torch

from climate_wm.models.energy_corrector import CorrectorConfig
from climate_wm.models.physics_world_model import CP, GRAV, L_V, PhysicsModelConfig, PhysicsWorldModel
from climate_wm.train.memory_filter import FilterConfig, expand_state_dict, train_filter
from test_physics_core import _model, _tiny


def _with_corrector(layout, stats, data, **corr):
    cfg = PhysicsModelConfig.from_dict({
        "core": {"trunc": 5, "nlat": 8, "nlon": 16, "dt_minutes": 60.0},
        "column": {"enabled": True, "hidden": 16, "layers": 1, "call_minutes": 120.0, "budget": True,
                   "energy_max_flux": 10.0, "energy_no_work": True},
        "corrector": dict({"enabled": True, "kind": "mlp", "hidden": 8, "layers": 1}, **corr)})
    torch.manual_seed(0)
    model = PhysicsWorldModel(cfg, layout, data.lat, data.lon)
    model.set_normalization(stats)
    return model


def _randomise(model):
    with torch.no_grad():
        for m in model.column.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.5)
                torch.nn.init.normal_(m.bias, std=0.5)
        for m in model.corrector.net.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight, std=0.5)
                torch.nn.init.normal_(m.bias, std=0.5)


def test_corrector_starts_at_zero_and_projects():
    layout, stats, data = _tiny()
    base = _model(layout, stats, data)
    model = _with_corrector(layout, stats, data)
    assert len(expand_state_dict(model, base.state_dict())) == 0  # the corrector is the only new module
    batch = {k: v[None] for k, v in data[0].items()}
    with torch.no_grad():
        model.cfg.column.energy_max_flux, model.cfg.column.energy_no_work = None, False
        assert torch.allclose(model.rollout(batch, 1), base.rollout(batch, 1), atol=1e-6)  # zero at the start
        model.cfg.column.energy_max_flux, model.cfg.column.energy_no_work = 10.0, True
    _randomise(model)
    state, ctx = model.begin(batch)
    forc = model.forcing_grid_at(ctx, 0)
    x, u, v = model.column_inputs(state, ctx["static_g"], forc)
    model.corrector.track = []
    out = model.column_forcing(state, ctx["static_g"], forc)
    dq, ds, dfu, dfv = model.corrector.correction(out, x, u, v, model.dp_layer)
    assert dq.abs().max() > 0 and dfu.abs().max() > 0
    dp = model.dp_layer[None, :, None, None]
    e = (CP * dq * dp).sum(1) / GRAV + (L_V * ds * dp).sum(1) / GRAV
    scale = ((CP * dq.abs() * dp).sum(1) / GRAV).max()
    assert e.abs().max() < 1e-5 * scale                                      # energy neutral (float32)
    fscale = ((dfu.abs() * dp).sum(1) / GRAV).max()
    assert ((dfu * dp).sum(1) / GRAV).abs().max() < 1e-5 * fscale             # momentum neutral
    work = ((dfu * u + dfv * v) * dp).sum(1) / GRAV
    assert (work <= 1e-5 * fscale * u.abs().max()).all()                      # no work
    flux = (CP * out["Q"] * dp).sum(1) / GRAV + (L_V * out["S"] * dp).sum(1) / GRAV
    assert (flux.abs() <= 10.0 + 1e-2).all()                                 # the E1 bound survives
    assert len(model.corrector.track) == 1 and model.corrector.track[0].shape == (1, 2)


def test_soft_corrector_collects_penalty_and_trains():
    layout, stats, data = _tiny()
    base = _model(layout, stats, data)
    model = _with_corrector(layout, stats, data, neutral=False, soft_weight=1.0, soft_bound=1.0)
    model.cfg.column.energy_max_flux, model.cfg.column.energy_no_work = None, False
    expand_state_dict(model, base.state_dict())
    _randomise(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    cfg = FilterConfig(fill_steps=1, free_steps=1, epochs=1, warmup_windows=1, train_model=False,
                       new_prefix="corrector.", log_every=1, save_every=1, cap=1e9)
    hist = train_filter(model, data, [0, 1], cfg, "cpu")
    assert len(hist["loss"]) == 2 and all(h == 0 for h in hist["blowups"])
    for k, v in model.state_dict().items():
        if k.startswith("corrector."):
            assert not torch.equal(v, before[k]), k
        else:
            assert torch.equal(v, before[k]), k  # the base stays frozen
