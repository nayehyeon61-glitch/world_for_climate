import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from climate_wm.models.tendency_control import (TendencyControlConfig, TendencyController, controller_state,
                                                load_controller)
from climate_wm.train.physics_game import play_physics
from climate_wm.train.prediction_game import GameConfig, episode_masks, evaluate_game, train_controller
from test_physics_core import _model, _tiny


@pytest.fixture(scope="module")
def setup():
    layout, stats, data = _tiny()
    model = _model(layout, stats, data, column=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return layout, data, model


def _ctrl(model, **kw):
    cfg = TendencyControlConfig(max_degree=2, layers=((500, 500), (850, 850)), hidden=8, **kw)
    torch.manual_seed(0)
    return TendencyController(model, cfg)


def test_controller_actions_move_the_low_harmonics(setup):
    layout, data, model = setup
    ctrl = _ctrl(model)
    assert ctrl.n_coef == 9 and ctrl.n_action == 9 * (3 * 2 + 1)
    batch = {k: v[None] for k, v in data[0].items()}
    state, _ = model.begin(batch)
    action = torch.zeros(1, ctrl.n_action)
    action[0, 0] = 1.0  # (n, m) = (0, 0) of temperature in the first layer (500 hPa)
    out = ctrl.apply(state, action)
    d = (out["temp"] - state["temp"]).abs()
    assert d[0, 0, 0, 0] > 0 and d[0, 1].max() == 0 and d[0, 0].sum() == d[0, 0, 0, 0]
    assert torch.equal(out["zeta"], state["zeta"])
    # zero-initialised controller: the controlled game equals the uncontrolled one
    gen = torch.Generator().manual_seed(0)
    masks, _ = episode_masks(layout, 1, GameConfig(), data.lat, data.lon, gen)
    h1, a1 = play_physics(model, ctrl, batch, masks, 2, obs_prob=0.5, gen=torch.Generator().manual_seed(1))
    h0, a0 = play_physics(model, ctrl, batch, masks, 2, control=False, obs_prob=0.5,
                          gen=torch.Generator().manual_seed(1))
    assert torch.allclose(h1, h0) and torch.allclose(a1, a0) and (h0 > 0).all()


@pytest.mark.parametrize("mode", ["rl", "grad"])
def test_physics_game_training_and_eval(setup, mode, tmp_path):
    layout, data, model = setup
    ctrl = _ctrl(model)
    ctrl.cost_weight = 1e-4
    cfg = GameConfig(mode=mode, steps=2, batch_size=1, members=2, horizon=2, obs_prob=0.5, log_every=1)
    loader = DataLoader(data, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(0))
    before = [p.clone() for p in ctrl.parameters()]
    hist = train_controller(model, ctrl, loader, cfg, play_fn=play_physics, log=lambda *_: None)
    assert len(hist) == 2 and all(np.isfinite(h["loss"]) for h in hist)
    assert any(not torch.equal(a, b) for a, b in zip(before, ctrl.parameters()))
    saved = controller_state(ctrl)
    again = load_controller(model, saved)
    assert all(torch.equal(a, b) for a, b in zip(ctrl.state_dict().values(), again.state_dict().values()))
    res = evaluate_game(model, again, data, [0, 1], ["none", "region"], 2, obs_prob=0.5,
                        play_fn=play_physics)
    ch = res["kinds"]["region"]["change"]
    assert {"control_hidden", "random_control_hidden", "no_diag_hidden"} <= set(ch)
