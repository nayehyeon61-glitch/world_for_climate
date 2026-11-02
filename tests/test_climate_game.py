import json
import os

import numpy as np
import torch

from climate_wm.models import knobs as K
from climate_wm.train.climate_game import (CEM, GameConfig, Scorer, Windows, calibrate, evaluate, grad_loss,
                                           load_policy, obs_dim, play, sparse_rewards, train_agent, Actor)
from test_physics_core import _model, _tiny


def _setup():
    layout, stats, data = _tiny()
    model = _model(layout, stats, data)
    torch.manual_seed(3)
    with torch.no_grad():   # the column starts at zero (N0); give it outputs for the knobs to scale
        model.column[-1].weight.normal_(0, 0.05)
    model.eval()
    return model, Windows(data)


def test_knobs_default_is_the_model_and_each_knob_acts():
    model, win = _setup()
    batch = win.batch([0])
    with torch.no_grad():
        ref = model.rollout(batch, 2)
        with K.knobs(model, torch.zeros(1, len(K.KNOBS))):
            same = model.rollout(batch, 2)
        assert torch.allclose(same, ref, atol=1e-5)
        assert model.column_scale is None and model.core.diffusion_power is None   # cleared
        for name in ("Q_lower", "S_lower", "F", "drag", "diffusion"):   # tiny levels: 500, 850 hPa
            a = torch.zeros(1, len(K.KNOBS))
            a[0, K.KNOBS.index(name)] = 1.0
            with K.knobs(model, a):
                moved = model.rollout(batch, 2)
            assert not torch.allclose(moved, ref, atol=1e-6), name
    st = K.settings(torch.tensor([[1.0] * 4 + [-1.0] * 3 + [0.0, 0.5, 1.0]]), [50, 300, 700, 1000])
    assert st["Q"].tolist() == [[1.5, 1.5, 1.5, 1.5]] and st["S"].tolist() == [[0.5, 0.5, 0.5, 0.5]]
    assert float(st["drag"]) == 1.25 and float(st["diffusion_power"]) == 0.5        # tau doubled
    assert K.group_mask(["momentum"]).tolist() == [0.0] * 7 + [1.0, 1.0, 0.0]


def test_play_rewards_blowups_and_sparse_delivery():
    model, win = _setup()
    sc = Scorer(model)
    cfg = GameConfig(days=1, steps_per_day=2)
    batch = win.batch([0, 1])
    zero = lambda o, d: torch.zeros(o.shape[0], len(K.KNOBS))
    ep = play(model, batch, zero, cfg, sc)
    assert ep["obs"].shape == (1, 2, obs_dim(sc)) and ep["rew"].shape == (1, 2)
    assert torch.isfinite(ep["rew"]).all() and (ep["rew"] < 0).all() and ep["done"].all()
    assert set(ep["bias"]) >= {"temperature", "2m_temperature", "all"} and (ep["blown_day"] == -1).all()
    # the terminal day adds the episode distance; the reference scales the reward
    ep2 = play(model, batch, zero, cfg, sc, ref={"day": [2.0], "term": 2.0})
    assert torch.allclose(ep2["rew"], ep["rew"] / 2, atol=1e-6)
    boom = play(model, batch, zero, GameConfig(days=1, steps_per_day=2, cap=-1.0), sc)
    assert (boom["blown_day"] == 0).all() and torch.equal(boom["rew"], torch.full((1, 2), -10.0))
    rew = torch.tensor([[-1.0], [-2.0], [-3.0], [-4.0], [-5.0]])
    done = torch.tensor([[False], [False], [False], [True], [False]])
    assert sparse_rewards(rew, done, 2)[:, 0].tolist() == [0.0, -3.0, 0.0, -7.0, 0.0]


def test_agents_train_resume_and_evaluate(tmp_path):
    model, win = _setup()
    sc = Scorer(model)
    cfg = GameConfig(days=1, steps_per_day=2)
    calib = calibrate(model, win, cfg, sc, n=2, batch=2, log=lambda s: None)
    assert len(calib["day"]) == 1 and len(calib["feat_mean"]) == 3 * sc.n and calib["blown"] == 0
    for kind in ("td3", "ppo", "cem"):
        out = str(tmp_path / kind)
        h = train_agent(kind, model, win, cfg, sc, calib, out, iters=2, batch=2, log=lambda s: None)["hist"]
        assert len(h) == 2 and os.path.exists(os.path.join(out, "agent.pt"))
        h = train_agent(kind, model, win, cfg, sc, calib, out, iters=3, batch=2, log=lambda s: None)["hist"]
        assert [r["iter"] for r in h] == [1, 2, 3]                                      # resumed, not restarted
    pol = load_policy("td3", str(tmp_path / "td3" / "agent.pt"), obs_dim(sc))
    res = evaluate(model, win, pol, cfg, sc, calib, str(tmp_path / "eval.json"), n=2, batch=2, log=lambda s: None)
    assert res["summary"]["inits"] == 2 and set(res["summary"]["knobs"]) == set(K.KNOBS)
    again = evaluate(model, win, pol, cfg, sc, calib, str(tmp_path / "eval.json"), n=2, batch=2, log=lambda s: None)
    assert len(again["rows"]) == 2                                                     # resumed from .progress
    const = load_policy("const", str(tmp_path / "cem" / "agent.pt"), obs_dim(sc))
    assert const(torch.zeros(3, obs_dim(sc)), 0).shape == (3, len(K.KNOBS))
    fixed = play(model, win.batch([0]), lambda o, d: torch.ones(1, len(K.KNOBS)),
                 GameConfig(days=1, steps_per_day=2, fixed_groups=("heating", "moistening", "momentum")), sc)
    assert fixed["act"].abs().sum() > 0          # the raw action is recorded; the held groups are masked in play


def test_grad_control_reaches_the_actor_and_cem_keeps_its_best(tmp_path):
    model, win = _setup()
    sc = Scorer(model)
    cfg = GameConfig(days=1, steps_per_day=2)
    for p in model.parameters():
        p.requires_grad_(False)
    actor = Actor(obs_dim(sc))
    with torch.enable_grad():
        loss = grad_loss(model, win.batch([0]), actor, cfg, sc)
        loss.backward()
    assert torch.isfinite(loss) and actor.net[-1].weight.grad.abs().sum() > 0
    calib = calibrate(model, win, cfg, sc, n=2, batch=2, log=lambda s: None)
    h = train_agent("grad", model, win, cfg, sc, calib, str(tmp_path / "grad"), iters=2, batch=1,
                    log=lambda s: None)["hist"]
    assert len(h) == 2 and all(np.isfinite(r["loss"]) for r in h)
    cem = CEM(seed=1)
    cand = cem.ask(6)
    assert torch.equal(cand[0], torch.zeros(len(K.KNOBS)))
    cem.tell(cand, torch.arange(6.0))
    assert torch.equal(cem.best[1], cand[5])
