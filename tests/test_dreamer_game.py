import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from climate_wm.models.physics_world_model import PhysicsModelConfig, PhysicsWorldModel
from climate_wm.models.tendency_control import TendencyControlConfig
from climate_wm.train.dreamer_game import (DreamerConfig, Heads, MemoryController, evaluate_memory_game,
                                           lambda_returns, play_memory, train_dreamer)
from climate_wm.train.physics_game import play_physics
from climate_wm.train.prediction_game import GameConfig, episode_masks
from test_physics_core import _tiny


def _mem_model(layout, stats, data, memory=None):
    cfg = PhysicsModelConfig.from_dict({
        "core": {"trunc": 5, "nlat": 8, "nlon": 16, "dt_minutes": 60.0},
        "column": {"enabled": True, "hidden": 16, "layers": 1, "call_minutes": 120.0},
        **({"memory": memory} if memory else {})})
    torch.manual_seed(0)
    model = PhysicsWorldModel(cfg, layout, data.lat, data.lon)
    model.set_normalization(stats)
    return model


@pytest.fixture(scope="module")
def setup():
    layout, stats, data = _tiny()
    base = _mem_model(layout, stats, data)
    model = _mem_model(layout, stats, data, {"channels": 4, "hidden": 8})
    missing = model.load_state_dict(base.state_dict(), strict=False).missing_keys
    assert missing and all(k.startswith("memory.") for k in missing)  # the stage 9 checkpoint loads
    return layout, data, base, model


def _agent(model):
    cfg = TendencyControlConfig(max_degree=2, layers=((500, 500), (850, 850)), hidden=8)
    torch.manual_seed(0)
    ctrl = MemoryController(model, cfg)
    ctrl.cost_weight = 1e-4
    return ctrl, Heads(ctrl.in_dim, ctrl.n_action, 16)


def test_untrained_memory_is_nudging(setup):
    layout, data, base, model = setup
    batch = {k: v[None] for k, v in data[0].items()}
    assert torch.allclose(model.rollout(batch, 2), base.rollout(batch, 2), atol=1e-6)
    gen = torch.Generator().manual_seed(0)
    masks, _ = episode_masks(layout, 1, GameConfig(), data.lat, data.lon, gen)
    run = lambda m, rule: play_memory(m, None, batch, masks, 2, obs_prob=1.0, rule=rule,
                                      gen=torch.Generator().manual_seed(1))
    h_mem, a_mem = run(model, "posterior")
    h_nud, a_nud = run(base, 1.0)
    assert torch.allclose(h_mem, h_nud, atol=1e-5) and torch.allclose(a_mem, a_nud, atol=1e-5)
    # the game rule of stages 7/9 is play_physics's default
    from climate_wm.models.tendency_control import TendencyController
    ctrl = TendencyController(base, TendencyControlConfig(max_degree=2, layers=((500, 500),), hidden=4))
    h_game, _ = play_physics(base, ctrl, batch, masks, 2, control=False, obs_prob=1.0,
                             gen=torch.Generator().manual_seed(1))
    h_reset, _ = run(base, "reset")
    assert torch.allclose(h_game, h_reset, atol=1e-6)


def test_memory_absorbs_innovation_and_fills_hidden_points(setup):
    layout, data, base, model = setup
    batch = {k: v[None] for k, v in data[0].items()}
    state, ctx = model.begin(batch)
    state, _ = model.advance(state, ctx, 0)
    pred = model.predict(state, ctx, 0)
    t2d = layout.target_to_dynamic
    mask = torch.zeros(1, len(layout.dynamic), *pred.shape[-2:])
    mask[..., :4] = 1
    truth = batch["targets"][:, 0][:, t2d]
    with torch.no_grad():
        for net in (model.memory.gain, model.memory.read):
            net[-1].weight.normal_(0, 0.1)
    post = model.posterior(state, ctx, pred[:, t2d], truth, mask)
    assert not torch.allclose(post["mem"], state["mem"])  # the innovation is stored
    model.memory.knock = {"innovation"}
    assert torch.equal(model.posterior(state, ctx, pred[:, t2d], truth, mask)["mem"], state["mem"])
    model.memory.knock = set()
    # the masked snapshot of the first time enters the memory before the first step,
    # and the actor reads that observation
    ctrl, _ = _agent(model)
    keep = []
    play_memory(model, ctrl, batch, mask, 1, keep=keep)
    assert keep[0][0]["mem"].abs().sum() > 0 and keep[0][4] is not None
    other = ctrl.observation_features(2 * truth, mask)
    assert not torch.allclose(other, keep[0][4]) and other[0, -1] == 0
    with torch.no_grad():
        for net in (model.memory.gain, model.memory.read):
            net[-1].weight.zero_()


def test_lambda_returns():
    r = torch.tensor([[1.0], [1.0]])
    v = torch.tensor([[0.0], [0.0], [10.0]])
    ret = lambda_returns(r, v, gamma=0.5, lam=1.0)  # Monte Carlo with bootstrap at the end
    assert torch.allclose(ret[:, 0], torch.tensor([1 + 0.5 * (1 + 0.5 * 10), 1 + 0.5 * 10]))


@pytest.mark.parametrize("arm", ["D", "D-blind", "D-noactor", "D-markov", "D-fixed", "D-frozen"])
def test_dreamer_training_and_eval(setup, arm, tmp_path):
    layout, data, base, _ = setup
    memory = {"D-markov": None, "D-fixed": {"channels": 4, "hidden": 8, "adaptive": False}}.get(
        arm, {"channels": 4, "hidden": 8})
    model = _mem_model(layout, data.stats, data, memory)
    model.load_state_dict(base.state_dict(), strict=False)
    ctrl, heads = _agent(model)
    cfg = DreamerConfig(steps=2, batch_size=1, horizon=2, imagine_horizon=2, members=2, obs_prob=1.0,
                        log_every=1, warmup_steps=0, train_actor=arm != "D-noactor",
                        train_model=arm != "D-frozen", imagination="blind" if arm == "D-blind" else "snapshot")
    loader = DataLoader(data, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(0))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    saved = []
    hist = train_dreamer(model, ctrl, heads, loader, cfg, save=saved.append, save_every=1)
    assert len(hist) == 2 and np.isfinite(hist[-1]["loss_model"])
    changed = any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert changed == (arm != "D-frozen")
    if cfg.train_actor:
        assert np.isfinite(hist[-1]["loss_actor"]) and hist[-1]["return_scale"] is not None
    # resuming from the first save repeats the second iteration
    model2 = _mem_model(layout, data.stats, data, memory)
    ctrl2, heads2 = _agent(model2)
    hist2 = train_dreamer(model2, ctrl2, heads2, loader, cfg, progress=saved[0])
    assert len(hist2) == 2 and abs(hist2[1]["hidden"] - hist[1]["hidden"]) < 1e-5
    res = evaluate_memory_game(model, ctrl if cfg.train_actor else None, base, data, [0, 1], ["region"], 2,
                               obs_prob=1.0)
    ch = res["kinds"]["region"]["change"]
    assert "full_hidden" in ch and "game_rule_hidden" in ch and "nudging_hidden" not in ch
    if model.memory is not None and model.memory.cfg.adaptive:
        assert len(res["log10_tau_hours_zonal"]) == 8
        assert np.shape(res["log10_tau_hours_channel_p10_p50_p90"]) == (model.memory.cfg.channels, 3)


def test_blowups_are_cut_or_capped(setup, monkeypatch):
    layout, data, base, model = setup
    batch = {k: torch.cat([v[None], v[None]]) for k, v in data[0].items()}
    masks = torch.zeros(2, len(layout.dynamic), len(data.lat), len(data.lon))
    advance = model.advance

    def exploding(state, ctx, k):  # the second sample blows up in its second step
        state, precip = advance(state, ctx, k)
        if k == 1:
            state = dict(state, temp=state["temp"].clone())
            state["temp"][-1] = float("nan")
        return state, precip

    monkeypatch.setattr(model, "advance", exploding)
    with torch.no_grad():
        h, a = play_memory(model, None, batch, masks, 2, cap=10.0)
    assert torch.isfinite(h).all() and h[1, 1] == 10.0 and h[0, 1] < 10.0
    one = {k: v[1:] for k, v in batch.items()}
    h, a = play_memory(model, None, one, masks[1:], 2, truncate=True)
    assert h.shape == (1, 1) and torch.isfinite(h).all()
