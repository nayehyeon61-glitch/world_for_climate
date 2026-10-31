import copy

import numpy as np
import pytest
import torch

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.data.normalization import compute_stats
from climate_wm.data.synthetic import make_synthetic_era5
from climate_wm.eval import causes
from climate_wm.models.feature_control import (ControlConfig, FeatureController, controlled,
                                               controller_state, load_controller)
from climate_wm.models.world_model import ClimateWorldModel, WorldModelConfig
from climate_wm.train import prediction_game as game

TINY = {
    "mesh": {"mesh_splits": 2, "latent_splits": 2, "patch_splits": 0,
             "pyramid_levels": (1, 2), "pyramid_detail_level": 2},
    "gnn": {"hidden_dim": 16, "vector_dim": 4, "latent_scalars": 8, "latent_vectors": 2,
            "encoder_layers": 1, "latent_layers": 1, "decoder_layers": 1},
    "lie": {"layers": 1, "shell_coupling": "independent"},
    "diffusion": {"layers": 1, "cond_dim": 16, "num_steps_denoising": 2},
    "pyramid": {"level_weights": (1.0, 0.5), "skip_dropout": 0.0, "level_layers": 1},
    "transition": "pyramid_diffusion",
}


@pytest.fixture(scope="module")
def setup():
    layout = gm.ChannelLayout(gm.get_task("graphcast_13_no_w", (500, 850)))
    ds = make_synthetic_era5(num_days=4, num_lat=9, num_lon=16, levels=(500, 850))
    raw = GraphCastWindowDataset(ds, layout, target_steps=2)
    stats = compute_stats(raw, range(len(raw)))
    data = GraphCastWindowDataset(ds, layout, stats, target_steps=2)
    batch = torch.utils.data.default_collate([data[0], data[1]])
    torch.manual_seed(0)
    model = ClimateWorldModel(WorldModelConfig.from_dict(TINY), layout, data.lat, data.lon)
    model.set_normalization(stats)
    with torch.no_grad():  # a moving transition, so steering has something to change
        model.lie.velocity.weight.normal_(0, 0.1)
    model.eval()
    return layout, data, stats, batch, model


def _controller(model, data, stats, **kw):
    torch.manual_seed(1)
    return FeatureController(model, data.lat, data.lon, ControlConfig(**kw), stats)


def _randomise(ctrl, std=0.5):
    with torch.no_grad():
        ctrl.net[-1].weight.normal_(0, std)


def test_untrained_controller_leaves_the_model_unchanged(setup):
    layout, data, stats, batch, model = setup
    ctrl = _controller(model, data, stats)
    hist = model.encode_history(batch)
    f = batch["forcings_tgt"][:, 0]
    with torch.no_grad():
        plain = model.step(hist, f, sample=False).z
        with controlled(model, ctrl):
            steered = model.step(hist, f, sample=False).z
        assert torch.allclose(plain.h, steered.h) and torch.allclose(plain.v, steered.v)
        assert model.feature_control is None  # detached after the block
        assert not any(k.startswith("feature_control") for k in model.state_dict())
        feats = ctrl.features(model, hist[-1])
        assert feats.shape == (2, len(layout.shells), ctrl.num_patches, ctrl.in_dim)
        assert torch.isfinite(feats).all() and feats[..., ctrl.cs + ctrl.cv:][..., :7].abs().sum() > 0
        _randomise(ctrl)
        act = ctrl.mean(feats)
        z = ctrl.apply(hist[-1], act)
        assert torch.equal(z.v, hist[-1].v) and not torch.allclose(z.h, hist[-1].h)  # scalars only
        with controlled(model, ctrl):
            assert not torch.allclose(model.step(hist, f, sample=False).z.h, plain.h)
    # open loop: the action depends on shell and patch only
    open_ctrl = _controller(model, data, stats, use_state=False)
    _randomise(open_ctrl)
    with torch.no_grad():
        a1 = open_ctrl.mean(open_ctrl.features(model, hist[-1]))
        a2 = open_ctrl.mean(open_ctrl.features(model, hist[-2]))
    assert torch.allclose(a1, a2) and not torch.allclose(a1[:, 0], a1[:, 1])
    again = load_controller(model, data.lat, data.lon, controller_state(ctrl), stats)
    assert torch.allclose(again.mean(feats), ctrl.mean(feats))


def test_masks_and_scores(setup):
    layout, data, stats, batch, model = setup
    gen = torch.Generator().manual_seed(0)
    groups = game.variable_groups(layout)
    assert "surface" in groups and "specific_humidity" in groups
    m, label = game.sample_mask(layout, "variables", data.lat, data.lon, gen)
    name = label.split(":")[1]
    assert m[groups[name]].all() and m.sum() == len(groups[name]) * m.shape[1] * m.shape[2]
    m, label = game.sample_mask(layout, "shell", data.lat, data.lon, gen)
    assert label.startswith("shell:") and 0 < m[:, 0, 0].sum() < len(layout.dynamic)
    fracs = []
    for _ in range(50):
        m, _ = game.sample_mask(layout, "region", data.lat, data.lon, gen, 0.25)
        assert (m == m[:1]).all()  # every channel hidden at the same places
        w = np.cos(np.deg2rad(data.lat))[:, None] * np.ones(len(data.lon))
        fracs.append(float((m[0].numpy() * w).sum() / w.sum()))
    assert 0.15 < np.mean(fracs) < 0.35
    tm = game.target_mask(model, m[None])
    assert tm.shape[1] == len(layout.targets) and tm.sum() == m.sum()
    pred, target = torch.randn(2, len(layout.targets), 9, 16), torch.randn(2, len(layout.targets), 9, 16)
    none = torch.zeros_like(tm.expand(2, -1, -1, -1))
    hid, all_ = game.errors(model, pred, target, none)
    assert torch.allclose(hid, all_)
    assert torch.allclose(all_, model.loss_fn.per_channel(pred, target) @ model.loss_fn.channel_w
                          / (model.loss_fn.channel_w.sum() * model.loss_fn.lat_w.mean()), rtol=1e-4)
    hid, _ = game.errors(model, pred, pred + 2 * tm.expand(2, -1, -1, -1), tm.expand(2, -1, -1, -1))
    assert torch.allclose(hid, torch.full_like(hid, 4.0))


def test_play_and_train(setup):
    layout, data, stats, batch, model = setup
    ctrl = _controller(model, data, stats)
    cfg = game.GameConfig(steps=2, members=3, horizon=2, log_every=1)
    masks, labels = game.episode_masks(layout, 2, cfg, data.lat, data.lon, torch.Generator().manual_seed(0))
    assert len(labels) == 2 and masks.shape == (2, len(layout.dynamic), 9, 16)
    with torch.no_grad():
        h0, a0 = game.play(model, ctrl, batch, masks, 2, control=False)
        h1, a1 = game.play(model, ctrl, batch, masks, 2, members=3)  # zero controller: same scores
        assert h1.shape == (6, 2) and torch.allclose(h1[::3], h0) and torch.allclose(a1[1::3], a0)
        # observational dropout: shared draws, re-observed truth improves the score
        g = lambda: torch.Generator().manual_seed(0)
        ho, _ = game.play(model, ctrl, batch, masks, 2, control=False, obs_prob=1.0, gen=g())
        assert torch.allclose(ho[:, 0], h0[:, 0]) and not torch.allclose(ho[:, 1], h0[:, 1])
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True, drop_last=True)
    for mode in ("rl", "grad"):
        c = _controller(model, data, stats)
        start = [p.detach().clone() for p in c.parameters()]
        hist = game.train_controller(model, c, loader, game.GameConfig(mode=mode, steps=2, members=3,
                                                                       horizon=2, log_every=1, lr=1e-2), log=print)
        assert len(hist) == 2 and np.isfinite(hist[-1]["loss"]) and "gain_mean_policy" in hist[-1]
        assert any(not torch.equal(a, b) for a, b in zip(start, c.parameters()))
        assert c.explore == 0.0
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())  # the world model is frozen


def test_evaluation_and_cause_variants(setup):
    layout, data, stats, batch, model = setup
    ctrl = _controller(model, data, stats)
    _randomise(ctrl)
    res = game.evaluate_game(model, ctrl, data, [0, 1, 2], ["none", "region"], horizon=2)
    for kind in ("none", "region"):
        r = res["kinds"][kind]
        assert len(r["labels"]) == 3 and len(r["mse"]["control_hidden"]) == 2
        assert np.abs(r["change"]["control_hidden"]["mean"]).max() > 0
        assert {"random_control_hidden", "no_diag_all"} <= set(r["change"])
    assert np.allclose(res["kinds"]["none"]["mse"]["control_hidden"], res["kinds"]["none"]["mse"]["control_all"])
    with controlled(model, ctrl):
        variants = causes.available_variants(model)
        assert {"no_control", "random_control", "no_diag"} <= set(variants)
        out = causes.error_decomposition(model, data, [0, 1], stats, lead_steps=2,
                                         variants=["no_control", "random_control", "no_diag"])
        for name in ("no_control", "random_control", "no_diag"):
            assert np.abs(np.array(out["variant_change"][name]["mean"])).max() > 0
        assert ctrl.enabled and not ctrl.random_actions and not ctrl.zero_diagnostics
    assert "no_control" not in causes.available_variants(model)


def test_resume_matches_an_uninterrupted_run(setup):
    layout, data, stats, batch, model = setup
    cfg = game.GameConfig(mode="grad", steps=3, horizon=2, log_every=0, lr=1e-2)
    loader = lambda: torch.utils.data.DataLoader(data, batch_size=2, shuffle=True, drop_last=True,
                                                 generator=torch.Generator().manual_seed(0))
    saves = []
    full = _controller(model, data, stats)
    game.train_controller(model, full, loader(), cfg, log=print,
                          save=lambda s: saves.append(copy.deepcopy(s)), save_every=1)
    assert [len(s["history"]) for s in saves] == [1, 2, 3]
    resumed = _controller(model, data, stats)
    hist = game.train_controller(model, resumed, loader(), cfg, log=print, progress=saves[0])
    assert len(hist) == 3
    for a, b in zip(full.parameters(), resumed.parameters()):
        assert torch.allclose(a, b, atol=1e-6)
