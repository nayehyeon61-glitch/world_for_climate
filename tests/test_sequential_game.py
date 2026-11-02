import copy
import datetime as dt

import numpy as np
import torch

from climate_wm.sim.intervention_game import SITES, action_field, train_actions
from climate_wm.sim.intervention_game import test_actions as held_out_actions
from climate_wm.sim.sequential_game import (CONTEXT, N_FEAT, SeqConfig, SeqPolicy, action_features, combine,
                                            discounted, forced_response_eval, forced_target, site_kernels,
                                            train_sequential)
from climate_wm.sim.speedy import Intervention
from test_action_channels import _with_action
from test_physics_core import _tiny


class StubSeqWorld:
    """Windows of the tiny dataset; an SST patch warms the targets, a perturbed member adds noise."""

    def __init__(self, layout, stats, data, workdir):
        self.layout, self.data = layout, data
        self.lat, self.lon = np.asarray(data.lat), np.asarray(data.lon)
        self.sea = np.ones((len(self.lat), len(self.lon)))
        self.workdir = workdir
        self.segments = []

    def branch_points(self, start, end, days):
        return [dt.datetime(2018, 1, 1) + dt.timedelta(days=5 * i) for i in range(4)]

    def start(self, t0):
        return "head", "restart0"

    def next_head(self, ds, t):
        return "head"

    def segment(self, tag, start, days, restart_in, intervention, head, perturbation=None, restart_out=None):
        self.segments.append((tag, restart_in, intervention))
        if restart_out is not None:
            open(restart_out, "w").write(tag)
        return intervention, perturbation

    def branch(self, t0, action, days, tag="branch", perturbation=None):
        return action.intervention, perturbation

    def departure(self, head, t, kernels):
        return np.zeros(len(SITES), np.float32)

    def window(self, ds, init, steps, action, device="cpu"):
        iv, pert = ds
        batch = {k: v[None].clone() for k, v in self.data[0].items()}
        batch["targets"] = batch["targets"][:, :steps]
        f = action_field(iv, self.lat, self.lon, self.sea)
        batch["targets"] = batch["targets"] + 0.1 * torch.from_numpy(f[0])[None, None, None]
        if pert is not None:
            g = torch.Generator().manual_seed(pert.seed)
            batch["targets"] = batch["targets"] + 0.05 * torch.randn(batch["targets"].shape, generator=g)
        a = action_field(action.intervention, self.lat, self.lon, self.sea)
        batch["action"] = torch.from_numpy(a)[None]
        batch["solar_factor"] = torch.tensor([action.intervention.solar_factor])
        return batch


def test_interventions_accumulate_on_the_world():
    acts = train_actions()
    sst = [a for a in acts if a.site is not None][:2]
    solar = [a for a in acts if a.name.startswith("solar")][0]
    iv = combine(Intervention(conv_factor=1.5, solar_factor=1.01), sst + [solar])
    assert iv.conv_factor == 1.5 and len(iv.patches) == 2
    assert np.isclose(iv.solar_factor, 1.01 * solar.intervention.solar_factor)
    assert np.allclose(discounted([1.0, 0.0, 2.0], 0.5), [1.5, 1.0, 2.0])


def test_forced_target_keeps_the_signal_and_drops_the_noise():
    g = torch.Generator().manual_seed(0)
    signal = torch.zeros(2, 50)
    signal[:, :10] = 1.0
    diffs = signal[None] + 0.1 * torch.randn((4, 2, 50), generator=g)
    diffs[:, :, 10:] = torch.randn((4, 2, 40), generator=g)       # pure noise: mean ~ se
    target, se2 = forced_target(diffs)
    assert (target[:, :10] - 1).abs().max() < 0.3
    assert target[:, 10:].square().mean() < 0.5 * diffs.mean(0)[:, 10:].square().mean()   # ~40% of the noise power stays (4 members)
    one, none = forced_target(diffs[:1])
    assert torch.equal(one, diffs[0]) and float(none.abs().sum()) == 0.0


def test_state_conditioned_policy_learns_the_rewarded_feature():
    acts = train_actions()
    cfg = SeqConfig(policy_lr=1.0, entropy_weight=0.0)
    pol = SeqPolicy(cfg)
    rng = np.random.default_rng(0)
    for _ in range(200):   # reward = choosing where the student is wrong
        err = rng.normal(size=len(SITES)).astype(np.float32)
        phi = action_features(acts, err, np.zeros(len(SITES), np.float32), [], 0.0, {})
        a = int(rng.choice(len(acts), p=pol.probs(phi)))
        pol.update([(phi, a, float(phi[a, 0]), 0)])
    assert pol.w[0] > 0.5
    kernels = site_kernels(np.linspace(-80, 80, 9), np.arange(0, 360, 30.0))
    assert kernels.shape == (len(SITES), 9, 12) and np.allclose(kernels.sum((1, 2)), 1.0)


def test_board4_arms_run_and_the_forced_response_eval(tmp_path):
    layout, stats, data = _tiny()
    steps = min(2, data[0]["targets"].shape[0])
    for arm in ("policy", "random", "bandit", "single", "noctx"):
        world = StubSeqWorld(layout, stats, data, tmp_path / arm)
        model = _with_action(layout, stats, data)
        saved = []
        cfg = SeqConfig(arm=arm, episodes=2, decisions=2, train_steps=1, reward_steps=steps, members=3,
                        policy_lr=2.0)
        hist = train_sequential(model, world, cfg, save=lambda p: saved.append(copy.deepcopy(p)), log=lambda s: None)
        assert len(hist["action"]) == 4 and len(saved) == 2 and not any(hist["blown"])
        assert np.isfinite(hist["reward"]).all() and np.isfinite(hist["state_loss"]).all()
        assert all(r[1] in ("restart0", str(tmp_path / arm / "seq_restart_0.bin")) for r in world.segments)
        # the second decision of an episode carries on from the first one's restart, with both actions
        second = [r for r in world.segments if r[0] == "seq_t0"][1]
        assert second[1].endswith("seq_restart_0.bin") and second[2] != Intervention()
        w = np.asarray(saved[-1]["w"])
        if arm == "noctx":
            assert np.allclose(w[CONTEXT], 0.0) and not np.allclose(w, 0.0)
        if arm in ("random", "bandit"):
            assert np.allclose(w, 0.0)
        if arm == "policy":
            w0 = _with_action(layout, stats, data).column[0].weight.detach()
            moved = (model.column[0].weight.detach() - w0).abs()
            assert moved[:, -3:].mean() > 3 * moved[:, :-3].mean()
        if arm == "bandit":
            assert not np.allclose(saved[-1]["logits"], 0.0)
        # resume at an episode boundary gives the same number of decisions
        hist2 = train_sequential(model, world, cfg, state=saved[0], log=lambda s: None)
        assert len(hist2["action"]) == 4
    other = _with_action(layout, stats, data)
    res = forced_response_eval({"a": model, "b": other}, world, world.branch_points(None, None, 1)[:2],
                               held_out_actions()[:2], steps, members=2, log=lambda s: None)
    assert len(res["rows"]) == 4 and len(res["points_done"]) == 2
    assert all(np.isfinite(r[lab]["rel_error"]) for r in res["rows"] for lab in "ab")
    assert all(r["zero_error"] <= 1.0 for r in res["rows"])
    part = {**res, "rows": res["rows"][:2], "points_done": res["points_done"][:1]}
    again = forced_response_eval({"a": model, "b": other}, world, world.branch_points(None, None, 1)[:2],
                                 held_out_actions()[:2], steps, members=2, log=lambda s: None, done=part)
    assert [r["a"]["rel_error"] for r in again["rows"]] == [r["a"]["rel_error"] for r in res["rows"]]
    assert N_FEAT == 8
