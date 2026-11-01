import numpy as np
import pytest
import torch

from climate_wm.train.teacher import TEND_KEYS, TeacherConfig, evaluate_teacher, teacher_rollout, train_teacher
from test_action_channels import _with_action
from test_physics_core import _tiny


class StubEpisodes:
    """Windows of the tiny dataset with a fixed tendency target (the teacher)."""

    def __init__(self, data, n_lev):
        self.data, self.n_lev = data, n_lev
        self.specs = [{"name": "A000", "group": "A"}]
        b = data[0]
        g = torch.linspace(-1, 1, b["targets"].shape[-2])[:, None] * torch.ones(b["targets"].shape[-1])
        self.target = torch.stack([g * 1e-5] * (len(TEND_KEYS) * n_lev))   # [4L, H, W]

    def windows(self, steps, stride=1):
        return [(0, 0)]

    def tendency_std(self, seed=0):
        return np.full(len(TEND_KEYS) * self.n_lev, 1e-5, np.float32)

    def batch(self, e, j, steps, device="cpu"):
        batch = {k: v[None].clone() for k, v in self.data[0].items()}
        batch["targets"] = batch["targets"][:, :steps]
        batch["tendency"] = self.target[None, None].repeat(1, steps, 1, 1, 1)
        return batch


def test_tendency_tracking_and_teacher_training_reduce_the_tendency_error():
    layout, stats, data = _tiny()
    model = _with_action(layout, stats, data)
    n_lev = len(layout.task.pressure_levels)
    eps = StubEpisodes(data, n_lev)
    steps = min(2, data[0]["targets"].shape[0])
    std = torch.from_numpy(eps.tendency_std())
    cfg = TeacherConfig(curriculum=((1, 12),), lambda_tend=1.0, lr=3e-3, log_every=100)
    with torch.no_grad():
        ls0, lt0, blown = teacher_rollout(model, eps.batch(0, 0, steps), steps, std, cfg)
    assert not blown and model.last_tendency is None and not model.track_tendency
    saved = []
    hist = train_teacher(model, eps, dataclasses_replace(cfg, save_every=5), save=saved.append, log=lambda s: None)
    assert len(hist["update"]) == 12 and len(saved) == 2 and len(hist["tend_std"]) == 4 * n_lev
    with torch.no_grad():
        _, lt1, _ = teacher_rollout(model, eps.batch(0, 0, steps), steps, std, cfg)
    assert float(lt1) < float(lt0)
    ev = evaluate_teacher(model, eps, steps, 1, std, log=lambda s: None)
    assert ev["windows"] == 1 and len(ev["corr_mean"]["Q"]) == steps and ev["blown"] == 0


def test_state_only_arm_ignores_the_teacher():
    layout, stats, data = _tiny()
    model = _with_action(layout, stats, data)
    eps = StubEpisodes(data, len(layout.task.pressure_levels))
    hist = train_teacher(model, eps, TeacherConfig(curriculum=((1, 3),), lambda_tend=0.0, log_every=100),
                         log=lambda s: None)
    assert np.isfinite(hist["tend"]).all() and len(hist["state"]) == 3


def dataclasses_replace(cfg, **kw):
    import dataclasses
    return dataclasses.replace(cfg, **kw)


def test_f1_stepper_trains_without_tendencies_and_reloads():
    pytest.importorskip("neuralop")
    from climate_wm.models.neural_stepper import NeuralStepper, StepperConfig
    layout, stats, data = _tiny()
    torch.manual_seed(0)
    f1 = NeuralStepper(StepperConfig(hidden=8, layers=2, modes=4), layout, data.lat)
    f1.set_normalization(stats)
    eps = StubEpisodes(data, len(layout.task.pressure_levels))
    steps = min(2, data[0]["targets"].shape[0])
    std = torch.from_numpy(eps.tendency_std())
    cfg = TeacherConfig(curriculum=((steps, 15),), lambda_tend=0.0, lr=3e-3, log_every=100)
    with torch.no_grad():
        ls0, lt0, blown = teacher_rollout(f1, eps.batch(0, 0, steps), steps, std, cfg)
        # starts as persistence of the last input
        state, ctx = f1.begin(eps.batch(0, 0, steps))
        assert torch.equal(f1.advance(state, ctx, 0)[0]["x"], state["x"])
    assert not blown and float(lt0) == 0.0
    hist = train_teacher(f1, eps, cfg, log=lambda s: None)
    with torch.no_grad():
        ls1, _, _ = teacher_rollout(f1, eps.batch(0, 0, steps), steps, std, cfg)
    assert float(ls1) < float(ls0) and len(hist["state"]) == 15
    ev = evaluate_teacher(f1, eps, steps, 1, std, log=lambda s: None)
    assert ev["windows"] == 1 and ev["blown"] == 0 and np.isnan(ev["corr_mean"]["Q"]).all()
    g = NeuralStepper(StepperConfig(hidden=8, layers=2, modes=4), layout, data.lat)
    res = g.load_state_dict(f1.state_dict(), strict=False)
    assert not [k for k in res.missing_keys if not k.endswith("_metadata")]
    with torch.no_grad():
        assert torch.allclose(g.forward_rollout(eps.batch(0, 0, steps), steps),
                              f1.forward_rollout(eps.batch(0, 0, steps), steps))
