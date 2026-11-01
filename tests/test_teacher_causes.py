import datetime as dt

import numpy as np
import torch

from climate_wm.eval.teacher_causes import (CUTS, evaluate_teacher_causes, lat_weights, model_knockout,
                                            response_metrics)
from climate_wm.sim.speedy import Intervention
from test_physics_core import _tiny

CFG = {"core": {"trunc": 5, "nlat": 8, "nlon": 16, "dt_minutes": 60.0},
       "column": {"enabled": True, "hidden": 16, "layers": 1, "call_minutes": 120.0, "budget": True},
       "action_channels": 3}


def test_response_metrics_agree_with_themselves_and_flip_with_the_sign():
    torch.manual_seed(0)
    d = torch.randn(3, 9, 16)
    w = lat_weights(np.linspace(-80, 80, 9))
    same = response_metrics(d, d, [0, 1, 2], w)
    assert abs(same["corr"] - 1) < 1e-6 and same["sign"] == 1.0 and abs(same["ratio"] - 1) < 1e-6
    flip = response_metrics(d, -2 * d, [0, 1], w)
    assert abs(flip["corr"] + 1) < 1e-6 and flip["sign"] == 0.0 and abs(flip["ratio"] - 2) < 1e-6
    assert np.isnan(response_metrics(d, d * float("nan"), [0], w)["corr"])


def test_cut_scales_only_enter_the_namelist_when_set():
    assert "heat_scale" not in Intervention().namelist()
    assert "heat_scale = 0.0" in Intervention(heat_scale=0.0).namelist() and not Intervention(heat_scale=0.0).is_null()


class StubWorld:
    """The control is the tiny dataset; each cut shifts the targets by a fixed pattern."""

    def __init__(self, data):
        self.data = data

    def branch(self, t0, action, days, tag="branch"):
        return action

    def window(self, ds, t0, steps, action, device="cpu"):
        b = {k: v[None].clone() for k, v in self.data[0].items()}
        b["targets"] = b["targets"][:, :steps]
        iv = action.intervention
        if not iv.is_null():
            b["targets"] = b["targets"] + 0.1 * (1 - iv.heat_scale) - 0.1 * (1 - iv.moist_scale)
        b["action"] = torch.zeros(1, 3, *b["targets"].shape[-2:])
        b["solar_factor"] = torch.tensor([iv.solar_factor])
        return b


def test_evaluate_teacher_causes_runs_resumes_and_summarises():
    from climate_wm.eval import teacher_causes as tc
    from climate_wm.models.physics_world_model import PhysicsModelConfig, PhysicsWorldModel
    layout, stats, data = _tiny()
    torch.manual_seed(1)
    model = PhysicsWorldModel(PhysicsModelConfig.from_dict(CFG), layout, data.lat, data.lon)
    model.set_normalization(stats)
    iv, used = model_knockout(("no_momentum", "no_drag"), model)
    assert iv is not None and "no_momentum" in used
    steps = data[0]["targets"].shape[0]
    old = tc.LEADS
    tc.LEADS = (1, steps)
    try:
        cuts = {k: CUTS[k] for k in ("no_heating", "no_moistening")}
        pts = [dt.datetime(1984, 1, 6), dt.datetime(1984, 2, 5)]
        saved = []
        res = evaluate_teacher_causes(CFG, model.state_dict(), layout, data.lat, data.lon, StubWorld(data), pts,
                                      steps, 1, cuts=cuts, save=lambda r: saved.append(len(r)), log=lambda s: None)
        assert saved == [1, 2] and res["inits"] == 2 and set(res["cuts"]) == set(cuts)
        cell = res["cuts"]["no_heating"][f"thermodynamics@{steps}"]
        assert set(cell) >= {"corr", "sign", "ratio", "gain", "rms_true"} and cell["rms_true"] > 0
        again = evaluate_teacher_causes(CFG, model.state_dict(), layout, data.lat, data.lon, StubWorld(data), pts,
                                        steps, 1, cuts=cuts, done=res["rows"][:1], log=lambda s: None)
        assert again["rows"][0] == res["rows"][0] and again["inits"] == 2
    finally:
        tc.LEADS = old
