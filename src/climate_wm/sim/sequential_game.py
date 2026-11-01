"""Stage 14.2 board 4: the sequential, state-conditioned intervention game.

Answer to the limits of board 2 (감자 22:02 KST 11/01, "이에대한 개선을 해보죠"):

1. Sequential decisions. An episode is `decisions` segments of `seg_days`
   (SPEEDY restarts at 00 UTC). Before each segment the experimenter sees the
   state (the student's current error near each site, the world's departure
   from the nature run there, the patches already applied, the step) and adds
   one intervention; interventions accumulate, the world carries on from the
   intervened segment, and the policy is trained by REINFORCE on the
   discounted return.
2. Forced-response reward. Each decision runs `members` pairs (with and
   without the new action) from the same states (member 0 unperturbed, the
   others perturbed restarts); the target is the ensemble-mean response, so
   chaotic divergence is averaged out. The reward is the student's learning
   progress on that response (relative error before minus after its update).
   Arm `single` uses member 0 alone (the board 2 signal) for the reward and
   the student's response loss.
3. The student (N1-T) trains on 24 h windows: state loss on the intervened
   segment + `response_weight` times its relative response error.
4. Randomized worlds: each episode draws convolution time, drag, CO2 and the
   solar constant (board 3 ranges); the actions act on top of them.

Arms: policy (sequential, state-conditioned), random, bandit (board 2's
context-free experimenter with the immediate reward), single (policy with the
single-branch reward), noctx (policy with the state features zeroed).
Rules G1-G4 in README section 14.2.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import shutil
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr

from ..data.era5_lazy_loader import standardize
from ..train.physics_game import _finite, attach_loss
from ..train.prediction_game import errors
from .episodes import PERTURBATION
from .intervention_game import (PATCH_RADIUS, SITES, Action, Experimenter, SpeedyWorld, random_world,
                                train_actions)
from .speedy import Intervention, Perturbation, read_native, to_dataset, to_wb2

N_FEAT = 8
FEATURES = ("student_error", "novelty", "already_here", "t2m_departure", "step", "sst_plus", "sst_minus",
            "global")
CONTEXT = slice(0, 5)   # zeroed in arm noctx


@dataclasses.dataclass
class SeqConfig:
    arm: str = "policy"            # policy | random | bandit | single | noctx
    episodes: int = 30
    decisions: int = 4
    seg_days: int = 2
    members: int = 4               # member 0 = unperturbed, 3 perturbed
    train_steps: int = 4           # 24 h student window
    reward_steps: int = 8          # response horizon for the reward (the segment)
    response_weight: float = 0.1
    action_lr_mult: float = 10.0   # the action inputs of the column start at zero: they learn 10x faster
    reward_decay: float = 0.9      # running rms that normalises the rewards of policy and bandit
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup: int = 10
    gamma: float = 0.7
    policy_lr: float = 0.5
    entropy_weight: float = 0.01
    baseline_decay: float = 0.8
    randomize: bool = True
    train_start: str = "1982-01-01"
    train_end: str = "1983-12-31"
    cap: float = 10.0
    seed: int = 0


def combine(base: Intervention, actions: Sequence[Action]) -> Intervention:
    """The world's intervention after `actions` on top of `base` (factors multiply, patches add)."""
    solar, co2, patches = base.solar_factor, base.co2_factor, list(base.patches)
    for a in actions:
        solar *= a.intervention.solar_factor
        co2 *= a.intervention.co2_factor
        patches += list(a.intervention.patches)
    return dataclasses.replace(base, solar_factor=solar, co2_factor=co2, patches=tuple(patches))


def site_kernels(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Normalised Gaussian weights of each site's patch [n_sites, H, W] (area weighted)."""
    la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    out = []
    for plon, plat in SITES:
        c = np.sin(la) * np.sin(np.deg2rad(plat)) + np.cos(la) * np.cos(np.deg2rad(plat)) * np.cos(lo - np.deg2rad(plon))
        d = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
        k = np.exp(-(d / PATCH_RADIUS) ** 2) * np.cos(la)
        out.append(k / k.sum())
    return np.stack(out).astype(np.float32)


def action_features(acts: Sequence[Action], err_site: np.ndarray, dep_site: np.ndarray, applied: Sequence[int],
                    step: float, counts: Dict[str, int]) -> np.ndarray:
    """phi [n_actions, N_FEAT] (FEATURES): err_site, dep_site per site (z-scored / K)."""
    phi = np.zeros((len(acts), N_FEAT), np.float32)
    for i, a in enumerate(acts):
        phi[i, 1] = 1.0 / math.sqrt(1.0 + counts.get(a.name, 0))
        if a.site is None:
            phi[i, 7] = 1.0
            continue
        amp = a.intervention.patches[0][3]
        phi[i, 0] = err_site[a.site]
        phi[i, 2] = float(a.site in applied)
        phi[i, 3] = dep_site[a.site] * np.sign(amp)
        phi[i, 4] = step
        phi[i, 5 if amp > 0 else 6] = 1.0
    return phi


class SeqPolicy:
    """Softmax over actions with logits phi(a, s) . w; REINFORCE on the
    discounted return with a per-step EMA baseline and an entropy bonus."""

    def __init__(self, cfg: SeqConfig):
        self.cfg = cfg
        self.w = np.zeros(N_FEAT)
        self.baseline = np.zeros(cfg.decisions)

    def probs(self, phi: np.ndarray) -> np.ndarray:
        z = phi @ self.w
        p = np.exp(z - z.max())
        return p / p.sum()

    def update(self, steps: Sequence[Tuple[np.ndarray, int, float, int]]) -> None:
        """steps: (phi, chosen index, return G, decision index)."""
        grad = np.zeros_like(self.w)
        for phi, a, g, k in steps:
            p = self.probs(phi)
            adv = g - self.baseline[k]
            grad += adv * (phi[a] - p @ phi)
            ent = -(p * np.log(p + 1e-12)).sum()
            grad += self.cfg.entropy_weight * (phi.T @ (-p * (np.log(p + 1e-12) + ent)))
        self.w += self.cfg.policy_lr * grad / max(len(steps), 1)
        for _, _, g, k in steps:
            self.baseline[k] = self.cfg.baseline_decay * self.baseline[k] + (1 - self.cfg.baseline_decay) * g


def discounted(rewards: Sequence[float], gamma: float) -> List[float]:
    out, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        out.append(g)
    return out[::-1]


class SequentialWorld:
    """Board 4's view of a SpeedyWorld: segments that carry on from the restart the
    last segment wrote, with the main trajectory's last two frames as input frames."""

    def __init__(self, world: SpeedyWorld):
        self.world = world
        self.layout, self.lat, self.lon = world.layout, world.lat, world.lon
        self.workdir = world.runner.workdir
        self.six = np.timedelta64(6, "h")

    def branch_points(self, start: str, end: str, days: int) -> List[dt.datetime]:
        return self.world.branch_points(start, end, days)

    def branch(self, *args, **kw) -> xr.Dataset:
        return self.world.branch(*args, **kw)

    def window(self, *args, **kw) -> Dict[str, torch.Tensor]:
        return self.world.window(*args, **kw)

    def start(self, t0: dt.datetime) -> Tuple[xr.Dataset, str]:
        """Input frames (t0 - 6 h, t0) and restart of the nature run at a branch point."""
        head = self.world.nature.sel(time=slice(np.datetime64(t0) - self.six, np.datetime64(t0)))
        return head, str(self.world.root / "restarts" / f"{t0:%Y%m%d}.bin")

    def next_head(self, ds: xr.Dataset, t: dt.datetime) -> xr.Dataset:
        return ds.sel(time=slice(np.datetime64(t) - self.six, np.datetime64(t)))

    def segment(self, tag: str, start: dt.datetime, days: int, restart_in: str, intervention: Intervention,
                head: xr.Dataset, perturbation: Optional[Perturbation] = None,
                restart_out: Optional[str] = None) -> xr.Dataset:
        """One SPEEDY segment from `restart_in` at `start`: `head` followed by its output;
        `restart_out` receives a copy of the end restart."""
        w = self.world
        run = w.runner.run(tag, start, start + dt.timedelta(days=days), restart_in=restart_in,
                           restart_out=restart_out is not None, intervention=intervention,
                           perturbation=perturbation)
        if restart_out is not None:
            shutil.copyfile(run / "restart.bin", restart_out)
        times = w.runner.output_times(run, start)
        ds = standardize(to_dataset(to_wb2(read_native(run, times), w.static_native), times), hours=None)
        if "level" in ds.dims:
            ds = ds.sel(level=w.levels)
        timed = [v for v in head.data_vars if "time" in head[v].dims and v in ds]
        merged = xr.concat([head[timed], ds[timed]], dim="time")
        for v in w.static_vars:
            merged[v] = w.nature[v]
        return merged

    def departure(self, head: xr.Dataset, t: dt.datetime, kernels: np.ndarray) -> np.ndarray:
        """2 m temperature of the world minus the nature run at t near each site (K)."""
        a = head["2m_temperature"].sel(time=np.datetime64(t)).transpose("lat", "lon").values
        b = self.world.nature["2m_temperature"].sel(time=np.datetime64(t)).transpose("lat", "lon").values
        return (kernels * (a - b)[None]).sum((1, 2)).astype(np.float32)


def free_run(model, batch, steps: int, cap: float) -> Tuple[List[torch.Tensor], bool]:
    """Normalised predictions over `steps` (with grad if enabled); blown = non-finite or
    the GraphCast error above `cap`."""
    attach_loss(model)
    state, ctx = model.begin(batch)
    preds = []
    tmask = torch.zeros_like(batch["targets"][:, 0])
    for k in range(steps):
        state, precip = model.advance(state, ctx, k)
        pred = model.predict(state, ctx, k, precip)
        _, e = errors(model, pred, batch["targets"][:, k], tmask)
        if not (torch.isfinite(e).all() and _finite(state).all()) or float(e.detach()) > cap:
            return preds, True
        preds.append(pred)
    return preds, False


def forced_target(diffs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """The forced response from member responses diffs [M, ...] (treatment - control from
    the same state): the ensemble mean shrunk towards 0 where it is not larger than its
    standard error (factor max(0, 1 - se^2 / mean^2) per point and channel), and the
    error variance se^2. One member: itself, no noise estimate (board 2's target).
    11/01 23:00: at 24-48 h the 3-member mean of an SST patch response has SNR 1.0 over
    the globe (chaotic divergence) and 2-5 near the patch; solar and CO2 1.0."""
    if diffs.shape[0] == 1:
        return diffs[0], torch.zeros_like(diffs[0])
    mean = diffs.mean(0)
    se2 = diffs.var(0, unbiased=True) / diffs.shape[0]
    keep = (1 - se2 / mean.square().clamp_min(1e-20)).clamp_min(0)
    return mean * keep, se2


def response_error(model, pred_t: Sequence[torch.Tensor], pred_c: Sequence[torch.Tensor],
                   target: torch.Tensor, se2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """GraphCast-weighted error of the predicted response (pred_t - pred_c) against the
    forced target [steps, C, H, W] over the first len(pred_t) steps, relative to the
    target's power plus its noise variance se2 (stable when the forced part is ~0)."""
    zero = torch.zeros_like(target[0])[None]
    num = den = 0.0
    for k, (pt, pc) in enumerate(zip(pred_t, pred_c)):
        num = num + errors(model, pt - pc, target[k][None], zero)[1].mean()
        den = den + float(errors(model, zero, target[k][None], zero)[1].mean())
        if se2 is not None:
            den = den + float(errors(model, zero, se2[k][None].sqrt(), zero)[1].mean())
    return num / max(den, 1e-12)


def action_input_params(model) -> Tuple[Optional[torch.nn.Parameter], int]:
    """The column's first-layer weight and its number of action input columns (the last ones)."""
    n = getattr(getattr(model, "cfg", None), "action_channels", 0) or 0
    for name, p in model.named_parameters():
        if name == "column.0.weight" and n:
            return p, n
    return None, 0


def site_errors(model, preds: Sequence[torch.Tensor], targets: torch.Tensor, kernels: np.ndarray) -> np.ndarray:
    """Student state error near each site (channel-weighted squared error averaged over
    the steps, kernel mean), z-scored across the sites."""
    lf = model.loss_fn
    with torch.no_grad():
        e = torch.stack([((p[0] - targets[0, k]).square() * lf.channel_w[:, None, None]).sum(0)
                         for k, p in enumerate(preds)]).mean(0).cpu().numpy()
    v = (kernels * e[None]).sum((1, 2))
    return ((v - v.mean()) / (v.std() + 1e-12)).astype(np.float32)


def member_perturbations(n: int, episode: int, k: int) -> List[Optional[Perturbation]]:
    """Perturbations of the members of one decision (member 0 = none; the same in every arm)."""
    return [None] + [Perturbation(seed=7000 + 1000 * episode + 10 * k + m, **PERTURBATION) for m in range(1, n)]


def train_sequential(model, world: SequentialWorld, cfg: SeqConfig, device="cpu", state: Optional[dict] = None,
                     save: Optional[Callable[[dict], None]] = None, log: Callable[[str], None] = print) -> dict:
    """Runs `cfg.episodes` episodes; `state` resumes at an episode boundary."""
    acts = train_actions()
    points = world.branch_points(cfg.train_start, cfg.train_end, cfg.decisions * cfg.seg_days)
    if not points:
        raise ValueError(f"no branch points in {cfg.train_start} .. {cfg.train_end}")
    world_rng = np.random.default_rng(cfg.seed)          # the same worlds in every arm
    act_rng = np.random.default_rng(cfg.seed + 1)
    kernels = site_kernels(world.lat, world.lon)
    params = [p for p in model.parameters() if p.requires_grad]
    n_updates = cfg.episodes * cfg.decisions
    w_in, n_act = action_input_params(model)
    groups = [{"params": [p for p in params if p is not w_in]}]
    if w_in is not None:
        groups.append({"params": [w_in], "lr": cfg.lr * cfg.action_lr_mult})
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    reward_ms = 0.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup, 1))
                                              * 0.5 * (1 + math.cos(math.pi * min(s / n_updates, 1.0))))
    policy, bandit = SeqPolicy(cfg), Experimenter(len(acts), cfg)
    counts: Dict[str, int] = {}
    hist = {k: [] for k in ("episode", "decision", "world", "action", "state_loss", "resp_before", "resp_after",
                            "reward", "snr", "kept", "probs_max", "blown", "seconds")}
    done = 0
    if state:
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        policy.w, policy.baseline = np.asarray(state["w"]), np.asarray(state["baseline"])
        bandit.logits, bandit.baseline = np.asarray(state["logits"]), state["bandit_baseline"]
        counts, hist = state["counts"], state["history"]
        reward_ms = state["reward_ms"]
        world_rng.bit_generator.state = state["world_rng"]
        act_rng.bit_generator.state = state["act_rng"]
        done = state["episodes_done"]
    t_start = time.time()
    work = world.workdir
    work.mkdir(parents=True, exist_ok=True)
    for ep in range(done, cfg.episodes):
        t0 = points[int(world_rng.integers(len(points)))]
        base = random_world(world_rng).intervention if cfg.randomize else Intervention()
        head, restart = world.start(t0)
        applied: List[Action] = []
        trace = []
        for k in range(cfg.decisions):
            tk = t0 + dt.timedelta(days=k * cfg.seg_days)
            prev = combine(base, applied)
            perts = member_perturbations(cfg.members, ep, k)
            # 1. controls (independent of the new action) and the student's view of them
            ctrl = [world.window(world.segment(f"seq_c{m}", tk, cfg.seg_days, restart, prev, head, pt), tk,
                                 cfg.reward_steps, Action("prev", prev), device) for m, pt in enumerate(perts)]
            model.eval()
            with torch.no_grad():
                pc, blown_c = free_run(model, ctrl[0], cfg.reward_steps, cfg.cap)
            err = site_errors(model, pc, ctrl[0]["targets"], kernels) if pc else np.zeros(len(SITES), np.float32)
            dep = world.departure(head, tk, kernels)
            phi = action_features(acts, err, dep, [a.site for a in applied if a.site is not None],
                                  k / max(cfg.decisions - 1, 1), counts)
            if cfg.arm == "noctx":
                phi[:, CONTEXT] = 0.0
            # 2. the experimenter's move
            if cfg.arm == "random":
                p = np.full(len(acts), 1.0 / len(acts))
            elif cfg.arm == "bandit":
                p = bandit.probs()
            else:
                p = policy.probs(phi)
            a = int(act_rng.choice(len(acts), p=p))
            act = acts[a]
            counts[act.name] = counts.get(act.name, 0) + 1
            now = combine(base, applied + [act])
            nxt = str(work / f"seq_restart_{k % 2}.bin")
            treat_ds = world.segment("seq_t0", tk, cfg.seg_days, restart, now, head, None, restart_out=nxt)
            treat = [world.window(treat_ds, tk, cfg.reward_steps, Action("now", now), device)]
            treat += [world.window(world.segment(f"seq_t{m}", tk, cfg.seg_days, restart, now, head, pt), tk,
                                   cfg.reward_steps, Action("now", now), device)
                      for m, pt in enumerate(perts) if m > 0]
            diffs = torch.stack([t["targets"][0] - c["targets"][0] for t, c in zip(treat, ctrl)])  # [M, S, C, H, W]
            r_ens, se2 = forced_target(diffs)
            late = slice(cfg.reward_steps // 2, cfg.reward_steps)
            mean = diffs.mean(0)
            snr = float(mean[late].square().mean().sqrt() / se2[late].mean().sqrt().clamp_min(1e-12))
            kept = float(r_ens.square().sum() / mean.square().sum().clamp_min(1e-20))   # forced share of the mean
            r_true, noise = (diffs[0], None) if cfg.arm == "single" else (r_ens, se2)
            # 3. the student: response error before, one update on the first 24 h, after
            with torch.no_grad():
                pt_, blown_t = free_run(model, treat[0], cfg.reward_steps, cfg.cap)
                before = (float(response_error(model, pt_, pc, r_true, noise))
                          if not (blown_t or blown_c) else float("nan"))
            model.train()
            opt.zero_grad(set_to_none=True)
            short_t = {**treat[0], "targets": treat[0]["targets"][:, :cfg.train_steps]}
            short_c = {**ctrl[0], "targets": ctrl[0]["targets"][:, :cfg.train_steps]}
            gt, blown_gt = free_run(model, short_t, cfg.train_steps, cfg.cap)
            gc, blown_gc = free_run(model, short_c, cfg.train_steps, cfg.cap)
            loss_s = float("nan")
            blown = blown_gt or blown_gc or blown_t or blown_c
            if not (blown_gt or blown_gc):
                tm = torch.zeros_like(short_t["targets"][:, 0])
                ls = torch.stack([errors(model, g, short_t["targets"][:, j], tm)[1].mean()
                                  for j, g in enumerate(gt)]).mean()
                loss = ls + cfg.response_weight * response_error(
                    model, gt, gc, r_true[:cfg.train_steps], None if noise is None else noise[:cfg.train_steps])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                old = w_in.detach()[:, :-n_act].clone() if w_in is not None else None
                opt.step()
                if w_in is not None:   # only the action columns take the larger step
                    with torch.no_grad():
                        w_in[:, :-n_act] = old + (w_in[:, :-n_act] - old) / cfg.action_lr_mult
                loss_s = float(ls.detach())
            sched.step()
            model.eval()
            with torch.no_grad():
                pt2, b1 = free_run(model, treat[0], cfg.reward_steps, cfg.cap)
                pc2, b2 = free_run(model, ctrl[0], cfg.reward_steps, cfg.cap)
                after = float(response_error(model, pt2, pc2, r_true, noise)) if not (b1 or b2) else float("nan")
            reward = (before - after) / max(before, 1e-6) if np.isfinite(before) and np.isfinite(after) else 0.0
            reward_ms = cfg.reward_decay * reward_ms + (1 - cfg.reward_decay) * reward ** 2 if reward_ms else reward ** 2
            scaled = reward / math.sqrt(reward_ms + 1e-24)
            if cfg.arm == "bandit":
                bandit.update(a, scaled)
            trace.append((phi, a, scaled, k))
            for key, v in (("episode", ep), ("decision", k), ("world", base.namelist().replace("\n", " ")),
                           ("action", act.name), ("state_loss", loss_s), ("resp_before", before),
                           ("resp_after", after), ("reward", reward), ("snr", snr), ("kept", kept), ("probs_max", float(p.max())),
                           ("blown", bool(blown)), ("seconds", time.time() - t_start)):
                hist[key].append(v)
            log(f"[seq {cfg.arm}] ep {ep + 1}/{cfg.episodes} d{k} {act.name}: state {loss_s:.4f} "
                f"resp {before:.3f}->{after:.3f} r {reward:+.4f} snr {snr:.1f} kept {kept:.2f} pmax {p.max():.3f} "
                f"({time.time() - t_start:.0f} s)")
            # 4. the world carries on from the intervened segment
            applied.append(act)
            restart = nxt
            head = world.next_head(treat_ds, tk + dt.timedelta(days=cfg.seg_days))
        if cfg.arm in ("policy", "single", "noctx"):
            g = discounted([r for _, _, r, _ in trace], cfg.gamma)
            policy.update([(phi, a, gk, k) for (phi, a, _, k), gk in zip(trace, g)])
        if save is not None:
            save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "w": policy.w.tolist(),
                  "baseline": policy.baseline.tolist(), "logits": bandit.logits.tolist(),
                  "bandit_baseline": bandit.baseline, "counts": counts, "history": hist, "reward_ms": reward_ms,
                  "world_rng": world_rng.bit_generator.state, "act_rng": act_rng.bit_generator.state,
                  "episodes_done": ep + 1})
    hist["policy_w"] = dict(zip(FEATURES, policy.w.tolist()))
    hist["bandit_probs"] = dict(zip([a.name for a in acts], bandit.probs().tolist()))
    return hist


@torch.no_grad()
def forced_response_eval(models, world: SequentialWorld, points: Sequence[dt.datetime], actions: Sequence[Action],
                         steps: int = 8, members: int = 5, device="cpu", log: Callable[[str], None] = print,
                         done: Optional[dict] = None, save: Optional[Callable[[dict], None]] = None) -> dict:
    """G1: per (branch point, action) in the default world, each model's error on the
    forced response (forced_target of `members` members from the nature state; member 0
    unperturbed; SPEEDY runs once for all models) relative to its power + noise, the same
    for a zero response (`zero_error`; skill = 1 - rel_error / zero_error), and the
    late-window pattern correlation with the forced target per target channel.
    `models`: one model or {label: model}; `done`/`save` resume per branch point."""
    if not isinstance(models, dict):
        models = {"model": models}
    for m in models.values():
        m.eval()
        attach_loss(m)
    keys = [c.key for c in world.layout.targets]
    lat_w = torch.from_numpy(np.cos(np.deg2rad(world.lat)).astype(np.float32))[:, None]
    null = Action("none", Intervention())
    days = int(math.ceil(steps / 4))
    res = done or {"rows": [], "keys": keys, "members": members, "steps": steps, "points_done": []}
    late = slice(steps // 2, steps)
    for i, t0 in enumerate(points):
        if t0.isoformat() in res["points_done"]:
            continue
        perts = [None] + [Perturbation(seed=9000 + 100 * i + m, **PERTURBATION) for m in range(1, members)]
        ctrl = [world.window(world.branch(t0, null, days, tag=f"fre_c{m}", perturbation=pt), t0, steps, null, device)
                for m, pt in enumerate(perts)]
        pcs = {lab: free_run(m, ctrl[0], steps, 1e9) for lab, m in models.items()}
        for act in actions:
            treat = [world.window(world.branch(t0, act, days, tag=f"fre_t{m}", perturbation=pt), t0, steps, act,
                                  device) for m, pt in enumerate(perts)]
            diffs = torch.stack([t["targets"][0] - c["targets"][0] for t, c in zip(treat, ctrl)])
            r, se2 = forced_target(diffs)
            first = next(iter(models.values()))
            zero = [torch.zeros_like(r[0])[None]] * steps
            row = {"init": t0.isoformat(), "action": act.name, "held_out": act.held_out,
                   "snr": float(diffs.mean(0)[late].square().mean().sqrt() / se2[late].mean().sqrt().clamp_min(1e-12)),
                   "zero_error": float(response_error(first, zero, zero, r, se2))}
            for lab, m in models.items():
                pc, blown_c = pcs[lab]
                pt_, blown_t = free_run(m, treat[0], steps, 1e9)
                out = {"rel_error": float("nan")}
                if not (blown_c or blown_t):
                    out["rel_error"] = float(response_error(m, pt_, pc, r, se2))
                    rp = torch.stack([a - b for a, b in zip(pt_, pc)])[:, 0]
                    mt, mp = r[late].mean(0), rp[late].mean(0)
                    for j, key in enumerate(keys):
                        a_ = mt[j] - (mt[j] * lat_w).sum() / (lat_w.sum() * mt.shape[-1])
                        b_ = mp[j] - (mp[j] * lat_w).sum() / (lat_w.sum() * mp.shape[-1])
                        den = float(((a_ * a_ * lat_w).sum() * (b_ * b_ * lat_w).sum()).sqrt())
                        out[key] = float((a_ * b_ * lat_w).sum() / den) if den > 0 else float("nan")
                row[lab] = out
            res["rows"].append(row)
            log(f"[fre] {t0:%Y-%m-%d} {act.name} (zero {row['zero_error']:.3f}, snr {row['snr']:.1f}): "
                + " ".join(f"{lab} {row[lab]['rel_error']:.3f}" for lab in models))
        res["points_done"].append(t0.isoformat())
        if save is not None:
            save(res)
    return res
