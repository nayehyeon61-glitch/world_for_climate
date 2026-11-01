import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from climate_wm.train.active_obs import (ActiveConfig, ObservationPolicy, ObsPolicyConfig, Patches, choice_entropy,
                                         evaluate_active_climate, evaluate_active_obs, play_observation,
                                         sample_subset, subset_log_prob, train_policy)
from test_dreamer_game import _mem_model
from test_physics_core import _tiny


@pytest.fixture(scope="module")
def setup():
    layout, stats, data = _tiny()
    base = _mem_model(layout, stats, data)
    model = _mem_model(layout, stats, data, {"channels": 4, "hidden": 8})
    model.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():  # a memory with some effect, so its features are not zero
        for p in model.memory.parameters():
            p.add_(0.1 * torch.randn_like(p))
    patches = Patches(data.lat, data.lon, model.lat_g, model.lon_g, bands=2, sectors=4)
    policy = ObservationPolicy(layout, patches, ObsPolicyConfig(bands=2, sectors=4, budget=0.25, hidden=8,
                                                                 max_horizon=4))
    from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
    data = GraphCastWindowDataset(data.source, layout, data.stats, target_steps=4)  # 4 target steps
    return layout, data, base, model, policy


def test_patches_are_equal_area_and_masks_follow_the_choice(setup):
    layout, data, base, model, policy = setup
    pt = policy.patches
    assert pt.n == 8 and set(pt.id_data.flatten().tolist()) == set(range(8))
    area = pt.pool_data.sum(1)
    assert torch.allclose(area, torch.ones(8))  # patch means
    w = np.cos(np.deg2rad(data.lat))[:, None].repeat(len(data.lon), 1)
    share = np.array([w[pt.id_data.numpy() == p].sum() for p in range(8)]) / w.sum()
    assert np.allclose(share, 1 / 8, atol=0.06)  # equal area up to the coarse grid
    choice = torch.tensor([[0, 5], [3, 3]])
    mask = pt.hidden_mask(choice, 3)
    assert mask.shape == (2, 3, len(data.lat), len(data.lon))
    assert torch.equal(mask[0, 0], (~torch.isin(pt.id_data, torch.tensor([0, 5]))).float())
    assert pt.chosen(choice)[1].sum() == 1  # a repeated id counts once


def test_plackett_luce_sampling_and_log_prob():
    logits = torch.zeros(3, 6)
    c = sample_subset(logits, 2, torch.Generator().manual_seed(0))
    assert c.shape == (3, 2) and (c[:, 0] != c[:, 1]).all()
    lp = subset_log_prob(logits, c)
    assert torch.allclose(lp, torch.full((3,), np.log(1 / 6) + np.log(1 / 5)))
    assert torch.equal(sample_subset(torch.tensor([[0.0, 3.0, 1.0, 2.0]]), 2, deterministic=True),
                       torch.tensor([[1, 3]]))
    strong = torch.tensor([[10.0, 0.0, 0.0, 0.0]], requires_grad=True)
    subset_log_prob(strong, torch.tensor([[0, 1]])).sum().backward()
    assert strong.grad is not None and torch.isfinite(strong.grad).all()
    assert choice_entropy(torch.zeros(1, 4)).item() == pytest.approx(np.log(4))


def test_episode_under_every_rule(setup):
    layout, data, base, model, policy = setup
    batch = {k: v[None] for k, v in data[0].items()}
    gen = torch.Generator().manual_seed(0)
    for rule in ("policy", "random", "age", "readout", "tendency", "full", "none"):
        e, choices, info = play_observation(model, policy, batch, rule, horizon=2, obs_steps=2, members=2, gen=gen)
        assert e.shape == (2, 2) and torch.isfinite(e).all()
        if rule in ("full", "none"):
            assert choices == []
        else:
            assert len(choices) == 1 and choices[0].shape == (2, 2)  # n = 25% of 8 patches; one choice before the end
            assert (choices[0][:, 0] != choices[0][:, 1]).all()
            assert info["tau"].shape == (2, 8) and torch.isfinite(info["tau"]).all()
    # the stage 9 model (no memory) plays the same game with nudging and zero memory features
    e, choices, info = play_observation(base, policy, batch, "policy", horizon=3, obs_steps=1, members=1, gen=gen)
    assert e.shape == (1, 3) and len(choices) == 1 and "tau" not in info
    # the age rule observes the patches observed longest ago
    e, choices, _ = play_observation(model, policy, batch, "age", horizon=3, obs_steps=3, members=1, gen=gen)
    assert set(choices[0][0].tolist()).isdisjoint(choices[1][0].tolist())


def test_training_resumes_and_evaluates(setup, tmp_path):
    layout, data, base, model, policy = setup
    loader = DataLoader(data, batch_size=1, shuffle=False)
    cfg = ActiveConfig(steps=3, members=4, horizon=3, obs_steps=2, budget=0.25, warmup_steps=1, log_every=1)
    saved = {}
    hist = train_policy(model, policy, loader, cfg, log=lambda *_: None, save=lambda p: saved.update(p), save_every=2)
    assert len(hist) == 3 and all(np.isfinite(h["loss"]) for h in hist) and "grad" in hist[0]
    assert sum(saved["counts"]) == 3 * 2 * 4 * 2  # steps x choices x members x n
    # resuming from the saved progress continues the same history
    policy2 = ObservationPolicy(layout, policy.patches, policy.cfg)
    hist2 = train_policy(model, policy2, loader, cfg, log=lambda *_: None, progress=saved)
    assert len(hist2) == 3 and hist2[1]["loss"] == hist[1]["loss"]
    res = evaluate_active_obs(model, policy, base, data, [0, 1], data.stats, rules=("policy", "random", "full", "none"),
                              budgets=(0.25, 0.5), horizon=3, obs_steps=2, base_rules=("random", "none"),
                              progress_path=str(tmp_path / "g.pt"))
    names = set(res["runs"])
    assert {"memory:policy", "memory:random", "memory:policy@0.5", "memory:full", "memory:none", "n1:random", "n1:none"} <= names
    assert "n1:random@0.5" in names and "memory:full@0.5" not in names
    r = res["runs"]["memory:policy"]
    assert len(r["mse"]) == 3 and set(r["rmse"]) == set(res["keys"]) and abs(sum(r["freq"]) - 2) < 1e-6
    assert len(res["tau_patch"]) == 8 and "memory:policy" in res["change"] and "memory:policy" in res["spearman_tau"]
    again = evaluate_active_obs(model, policy, base, data, [0, 1], data.stats, rules=("policy", "random", "full", "none"),
                                budgets=(0.25, 0.5), horizon=3, obs_steps=2, base_rules=("random", "none"),
                                progress_path=str(tmp_path / "g.pt"))
    assert again["runs"]["memory:policy"]["mse"] == r["mse"]


def test_climate_frame(setup, tmp_path):
    from climate_wm.eval.climatology import Climatology

    layout, data, base, model, policy = setup
    long = data
    clim = Climatology.from_window_dataset(long, range(len(long)), 1)
    res = evaluate_active_climate(model, policy, base, long, [0, 1], clim, long.stats, rules=("policy", "random", "full"),
                                  spinup=2, weeks=2, week=1, base_rules=("random",), progress_path=str(tmp_path / "c.pt"))
    assert set(res["runs"]) == {"climatology", "memory:policy", "memory:random", "memory:full", "n1:random"}
    r = res["runs"]["memory:policy"]
    assert len(r["all"]) == 2 and len(r["freq"]) == 8 and r["blowups"] == 0
    assert set(res["change"]) == {"climatology", "memory:policy", "memory:full", "n1:random"}
