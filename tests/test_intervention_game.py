import datetime as dt

import numpy as np
import torch

from climate_wm.sim.intervention_game import (ACTION_SCALE, HELD_OUT_SITES, SITES, Experimenter, GameConfig,
                                              action_field, all_actions, response_eval, test_actions,
                                              train_actions, train_game)
from climate_wm.sim.speedy import Intervention
from test_action_channels import _with_action
from test_physics_core import _tiny


class StubWorld:
    """Windows of the tiny dataset; an intervention adds a warm offset to the targets."""

    def __init__(self, layout, stats, data):
        self.layout, self.stats, self.data = layout, stats, data
        self.lat = np.asarray(data.lat)
        self.lon = np.asarray(data.lon)
        self.sea = np.ones((len(self.lat), len(self.lon)))

    def branch_points(self, start, end, days):
        return [dt.datetime(2018, 1, 1) + dt.timedelta(days=5 * i) for i in range(4)]

    def branch(self, t0, action, days, tag="branch"):
        return action

    def window(self, ds, init, steps, action, device="cpu"):
        batch = {k: v[None].clone() for k, v in self.data[0].items()}
        batch["targets"] = batch["targets"][:, :steps]
        f = action_field(action.intervention, self.lat, self.lon, self.sea)
        batch["targets"] = batch["targets"] + 0.1 * torch.from_numpy(f[0])[None, None, None]
        batch["action"] = torch.from_numpy(f)[None]
        batch["solar_factor"] = torch.tensor([action.intervention.solar_factor])
        return batch


def test_action_sets_and_fields():
    acts = all_actions()
    assert len(acts) == 2 * len(SITES) + 4
    assert not any(a.held_out for a in train_actions())
    assert sum(a.held_out for a in test_actions()) == 2 * len(HELD_OUT_SITES)
    lat, lon = np.linspace(-80, 80, 9), np.arange(0, 360, 30.0)
    f = action_field(Intervention(patches=((180.0, 0.0, 12.0, 2.0),), co2_factor=1.25), lat, lon,
                     np.ones((9, 12)))
    assert np.isclose(f[0].max(), 2.0 * ACTION_SCALE[0]) and np.isclose(f[2, 0, 0], 0.25 * ACTION_SCALE[2])
    assert f[0][4, 6] == f[0].max()                    # centred on (180 E, 0 N)


def test_experimenter_moves_towards_rewarded_action():
    ex = Experimenter(4, GameConfig())
    for _ in range(30):
        ex.update(2, 1.0)
        ex.update(0, 0.0)
    assert ex.probs().argmax() == 2 and ex.entropy() < np.log(4)


def test_game_arms_run_and_response_eval():
    layout, stats, data = _tiny()
    world = StubWorld(layout, stats, data)
    steps = min(2, data[0]["targets"].shape[0])
    for arm in ("lp", "random", "nature"):
        model = _with_action(layout, stats, data)
        saved = []
        hist = train_game(model, world, GameConfig(arm=arm, updates=3, train_steps=steps, save_every=2,
                                                   probe_day=1), save=saved.append, log=lambda s: None)
        assert len(hist["update"]) == 3 and len(saved) == 2
        assert (arm == "lp") == np.isfinite(hist["reward"]).all()
    res = response_eval(model, world, world.branch_points(None, None, 1)[:1], test_actions()[:2], steps,
                        log=lambda s: None)
    assert len(res["rows"]) == 2 and "rmse" in res["rows"][0][res["keys"][0]]
