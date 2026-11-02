import numpy as np
import torch

from climate_wm.models.physics_world_model import PhysicsModelConfig, PhysicsWorldModel
from climate_wm.train.memory_filter import expand_state_dict
from test_physics_core import _model, _tiny


def _with_action(layout, stats, data, **extra):
    cfg = PhysicsModelConfig.from_dict(dict({
        "core": {"trunc": 5, "nlat": 8, "nlon": 16, "dt_minutes": 60.0},
        "column": {"enabled": True, "hidden": 16, "layers": 1, "call_minutes": 120.0, "budget": True},
        "action_channels": 3}, **extra))
    torch.manual_seed(1)
    model = PhysicsWorldModel(cfg, layout, data.lat, data.lon)
    model.set_normalization(stats)
    return model


def test_widened_checkpoint_ignores_zero_action_and_reacts_to_one():
    layout, stats, data = _tiny()
    base = _model(layout, stats, data)
    model = _with_action(layout, stats, data)
    expanded = expand_state_dict(model, base.state_dict())
    assert any(k.startswith("column.0") for k in expanded)
    batch = {k: v[None] for k, v in data[0].items()}
    with torch.no_grad():
        ref = base.rollout(batch, 2)
        assert torch.allclose(model.rollout(batch, 2), ref, atol=1e-6)        # no action key
        h, w = batch["static"].shape[-2:]
        batch["action"] = torch.zeros(1, 3, h, w)
        assert torch.allclose(model.rollout(batch, 2), ref, atol=1e-6)        # zero action
        model.column[0].weight[:, -3:] = 0.5                                  # learn a response
        torch.nn.init.normal_(model.column[-1].weight, std=0.1)
        base.column[-1].weight.copy_(model.column[-1].weight)
        ref = base.rollout({k: v for k, v in batch.items() if k != "action"}, 2)
        batch["action"][:, 0] = 1.0
        assert (model.rollout(batch, 2) - ref).abs().max() > 1e-4


def test_daily_mean_solar_and_solar_factor():
    layout, stats, data = _tiny()
    model = _with_action(layout, stats, data, solar="daily_mean")
    t0 = np.array([np.datetime64("2018-06-01T00:00", "s").astype(np.int64)])
    a = model._forcing_grid(t0)
    b = model._forcing_grid(t0 + 6 * 3600)
    k = model.forcing_keys.index("toa_incident_solar_radiation")
    assert torch.allclose(a[:, k], b[:, k])                                   # constant through the day
    c = model._forcing_grid(t0, solar_scale=1.02)
    std = model.forcing_std[k]
    phys_a = a[:, k] * std + model.forcing_mean[k]
    phys_c = c[:, k] * std + model.forcing_mean[k]
    assert torch.allclose(phys_c, phys_a * 1.02, rtol=1e-4, atol=1e-2)
