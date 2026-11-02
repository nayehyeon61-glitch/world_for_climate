import copy
import datetime as dt

import numpy as np
import torch

from climate_wm.eval.hidden_forcing import (HiddenConfig, fill_then_free, forced_signal, hidden_anomaly,
                                           hidden_forcing_eval, probe_r2)
from climate_wm.sim.intervention_game import action_field
from test_action_channels import _with_action
from test_physics_core import _tiny


class StubHiddenWorld:
    """Windows of the tiny dataset tiled to any length; a hidden SST anomaly warms every target persistently."""

    def __init__(self, layout, data):
        self.layout, self.data = layout, data
        self.lat, self.lon = np.asarray(data.lat), np.asarray(data.lon)
        self.sea = np.ones((len(self.lat), len(self.lon)))
        self.branches = []

    def branch(self, t0, action, days, tag="branch", perturbation=None):
        self.branches.append((t0, action.name, days))
        return action.intervention

    def window(self, ds, init, steps, action, device="cpu"):
        item = self.data[0]
        reps = -(-steps // item["targets"].shape[0])
        batch = {k: v[None].clone() for k, v in item.items()}
        for k in ("targets", "forcings_tgt"):
            batch[k] = batch[k].repeat(1, reps, 1, 1, 1)[:, :steps]
        f = torch.from_numpy(action_field(ds, self.lat, self.lon, self.sea)[0])
        batch["targets"] = batch["targets"] + 0.2 * f[None, None, None]
        batch["action"] = torch.from_numpy(action_field(action.intervention, self.lat, self.lon, self.sea))[None]
        batch["solar_factor"] = torch.tensor([action.intervention.solar_factor])
        return batch


def test_hidden_forcing_eval_rows_probe_and_resume():
    layout, stats, data = _tiny()
    mem = _with_action(layout, stats, data, memory={"channels": 4, "hidden": 8})
    torch.manual_seed(2)
    for p in mem.memory.parameters():
        p.data.normal_(0, 0.3)
    markov = _with_action(layout, stats, data)
    world = StubHiddenWorld(layout, data)
    cfg = HiddenConfig(histories_days=(0, 1), free_steps=2, cap=1e6)
    iv = hidden_anomaly(np.random.default_rng(0), cfg)
    assert len(iv.patches) == 3 and len({p[:2] for p in iv.patches}) == 3

    batch = world.window(iv, None, 6, type("A", (), {"intervention": iv})(), "cpu")
    with torch.no_grad():
        p_mem, m_after, blown = fill_then_free(mem, batch, 4, 2)
        p_frozen, m_frozen, _ = fill_then_free(mem, batch, 4, 2, knock=("frozen",))
    assert not blown and m_after is not None and p_mem.shape[0] == 2
    assert not np.allclose(p_mem, p_frozen) and mem.memory.knock == set()   # the memory acts; the knock is reset
    assert torch.count_nonzero(m_frozen) == 0          # frozen from an empty start: m stays 0 (the Markov control)

    points = [dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 6)]
    saved = []
    res = hidden_forcing_eval({"mem": mem, "markov": markov}, world, points, cfg, shown={"mem"},
                              log=lambda s: None, save=lambda r: saved.append(copy.deepcopy(r)))
    # per point: anomaly mem 3 variants x 2 h + markov-model 2 h + mem shown 2 variants x 2 h; nature mem 6 + 2
    assert len(res["rows"]) == 2 * 20 and len(res["probe"]) == 2 * 4
    assert {r["variant"] for r in res["rows"] if r["shown"]} == {"memory", "frozen"}
    assert {r["variant"] for r in res["rows"] if r["model"] == "markov"} == {"memory"}
    assert all(len(r["all"]) == 2 and len(r["t850"]) == 2 for r in res["rows"] if "t850" in r)
    assert all(np.isfinite(r["all"]).all() for r in res["rows"])
    r2 = probe_r2(res["probe"])
    assert set(r2) == {("mem", 0), ("mem", 1)}
    assert len(saved) == 2 * 3 and saved[0]["truths"] == ["2018-01-01T00:00:00|anomaly"]   # saved per truth
    for half in (saved[0], saved[1]):           # resume after the anomaly branch, and after the whole point
        again = hidden_forcing_eval({"mem": mem, "markov": markov}, world, points, cfg, shown={"mem"},
                                    log=lambda s: None, done=copy.deepcopy(half))
        assert len(again["rows"]) == len(res["rows"]) and again["points"] == res["points"]
        assert [(r["init"], r["truth"], r["variant"], r["h"]) for r in again["rows"]] == \
            [(r["init"], r["truth"], r["variant"], r["h"]) for r in res["rows"]]


def test_forced_signal_is_the_anomaly_footprint_in_the_truth():
    layout, stats, data = _tiny()
    model = _with_action(layout, stats, data)
    world = StubHiddenWorld(layout, data)
    cfg = HiddenConfig(histories_days=(0, 1), free_steps=2, cap=1e6)
    points = [dt.datetime(2018, 1, 1), dt.datetime(2018, 1, 6)]
    sig = forced_signal(model, world, points, cfg)
    assert [(r["init"][:10], r["h"]) for r in sig] == [("2018-01-01", 0), ("2018-01-01", 1),
                                                         ("2018-01-06", 0), ("2018-01-06", 1)]
    # the stub warms every target by 0.2 x the anomaly field: a positive, lead-independent signal
    assert all(len(r["all"]) == 2 and min(r["all"]) > 0 for r in sig)
    assert all(np.allclose(r["all"], r["all"][0]) for r in sig)
    # the anomalies are hidden_forcing_eval's (same seed): its first point's patches
    rng = np.random.default_rng(cfg.seed)
    assert sig[0]["patches"] == [list(p) for p in hidden_anomaly(rng, cfg).patches]
