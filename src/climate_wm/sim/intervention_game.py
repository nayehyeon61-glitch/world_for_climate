"""Stage 14 board 2: the intervention two-player game in the SPEEDY world.

The experimenter picks an intervention at a branch point of the nature run
(an SST patch, the solar constant or the CO2 band absorptivity); SPEEDY runs
the intervened branch; the world model, which sees the intervention through
its action channels (`PhysicsModelConfig.action_channels`), learns from the
branch. The experimenter's reward is the world model's learning progress on a
held-back part of the same branch (error before minus after the update), not
its raw error, so unlearnable noise is not rewarded.

Arms: `lp` (learning-progress experimenter), `random` (uniform over the same
actions), `nature` (no interventions, same number of updates; also the domain
adaptation phase `adapt`). Rules S1, S2, S4 in README section 14.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr

from ..data.era5_lazy_loader import GraphCastWindowDataset, standardize
from ..train.physics_game import _finite, attach_loss
from ..train.prediction_game import errors
from .speedy import Intervention, SpeedyRunner, read_native, read_static, to_dataset, to_wb2

# 24 open-ocean patch centres (lon E, lat N): tropical and subtropical Pacific,
# Atlantic and Indian oceans, and the Southern Ocean
SITES: Tuple[Tuple[float, float], ...] = (
    (180.0, 0.0), (210.0, 0.0), (240.0, -5.0), (150.0, 10.0), (200.0, 20.0), (230.0, -20.0),
    (180.0, -30.0), (165.0, 35.0), (205.0, 40.0), (250.0, 15.0), (330.0, 0.0), (320.0, 25.0),
    (340.0, -20.0), (310.0, 40.0), (345.0, -35.0), (70.0, -10.0), (85.0, 5.0), (90.0, -25.0),
    (60.0, -35.0), (65.0, 12.0), (120.0, -45.0), (20.0, -45.0), (270.0, -40.0), (300.0, -55.0))
HELD_OUT_SITES = tuple(range(0, len(SITES), 4))   # 6 sites never chosen in training
PATCH_RADIUS = 12.0                                 # degrees of arc (e-folding)
PATCH_AMPS = (-2.0, 2.0)                            # K
SOLAR = (0.98, 1.02)
CO2 = (0.8, 1.25)
ACTION_SCALE = (0.5, 50.0, 4.0)                     # channel = scale * (SST K, solar - 1, co2 - 1)


@dataclasses.dataclass(frozen=True)
class Action:
    name: str
    intervention: Intervention
    site: Optional[int] = None

    @property
    def held_out(self) -> bool:
        return self.site is not None and self.site in HELD_OUT_SITES


def all_actions() -> List[Action]:
    acts = []
    for i, (lon, lat) in enumerate(SITES):
        for amp in PATCH_AMPS:
            acts.append(Action(f"sst{i:02d}{'+' if amp > 0 else '-'}",
                               Intervention(patches=((lon, lat, PATCH_RADIUS, amp),)), i))
    acts += [Action(f"solar{f:.2f}", Intervention(solar_factor=f)) for f in SOLAR]
    acts += [Action(f"co2x{f:.2f}", Intervention(co2_factor=f)) for f in CO2]
    return acts


def train_actions() -> List[Action]:
    return [a for a in all_actions() if not a.held_out]


def test_actions() -> List[Action]:
    """Held-out SST sites (both signs) and the global forcings (seen in training:
    S4 asks for the right sign, not generalisation)."""
    return [a for a in all_actions() if a.held_out or a.site is None]


def action_field(iv: Intervention, lat: np.ndarray, lon: np.ndarray, sea: np.ndarray) -> np.ndarray:
    """Action channels [3, H, W] on the data grid (lat, lon degrees; sea fraction [H, W])."""
    la = np.deg2rad(lat)[:, None]
    lo = np.deg2rad(lon)[None, :]
    sst = np.zeros((len(lat), len(lon)))
    for plon, plat, rad, amp in iv.patches:
        c = (np.sin(la) * np.sin(np.deg2rad(plat))
             + np.cos(la) * np.cos(np.deg2rad(plat)) * np.cos(lo - np.deg2rad(plon)))
        d = np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
        sst += amp * np.exp(-(d / rad) ** 2)
    sst *= (sea > 0)
    out = np.stack([sst * ACTION_SCALE[0],
                    np.full_like(sst, (iv.solar_factor - 1.0) * ACTION_SCALE[1]),
                    np.full_like(sst, (iv.co2_factor - 1.0) * ACTION_SCALE[2])])
    return out.astype(np.float32)


class SpeedyWorld:
    """Branches of the SPEEDY nature run as window datasets for the world model."""

    def __init__(self, speedy_root: str, data_root: str, workdir: str, nature: xr.Dataset,
                 layout, stats, levels: Sequence[int]):
        self.runner = SpeedyRunner(speedy_root, workdir)
        self.root = Path(data_root)
        self.static_native = read_static(self.root / "static.nc")
        self.nature = nature            # standardised (pipeline.open_dataset)
        self.layout, self.stats, self.levels = layout, stats, list(levels)
        self.lat = nature.lat.values
        self.lon = nature.lon.values
        land = nature["land_sea_mask"].transpose("lat", "lon").values
        self.sea = 1.0 - land
        self.static_vars = [v for v in nature.data_vars if "time" not in nature[v].dims]

    def branch_points(self, start: str, end: str, days: int) -> List[dt.datetime]:
        """Restart dates in [start, end - days] whose t0 - 6 h frame is in the nature run."""
        lo, hi = dt.datetime.fromisoformat(start), dt.datetime.fromisoformat(end)
        first = self.nature.time.values[0]
        last = self.nature.time.values[-1]
        out = []
        for f in sorted((self.root / "restarts").glob("*.bin")):
            t = dt.datetime.strptime(f.stem, "%Y%m%d")
            t_end = np.datetime64(t + dt.timedelta(days=days))
            if lo <= t and t + dt.timedelta(days=days) <= hi + dt.timedelta(days=1) \
                    and np.datetime64(t) - np.timedelta64(6, "h") >= first and t_end <= last:
                out.append(t)
        return out

    def check_sites(self) -> None:
        for i, (lon, lat) in enumerate(SITES):
            j = int(np.abs(self.lat - lat).argmin())
            k = int(np.abs(((self.lon - lon) + 180) % 360 - 180).argmin())
            if self.sea[j, k] < 0.9:
                raise ValueError(f"site {i} ({lon}, {lat}) is not open sea ({self.sea[j, k]:.2f})")

    def branch(self, t0: dt.datetime, action: Action, days: int, tag: str = "branch") -> xr.Dataset:
        """Nature frames t0 - 6 h and t0 followed by the (intervened) branch to t0 + days."""
        six = np.timedelta64(6, "h")
        start = np.datetime64(t0) - six
        end = np.datetime64(t0 + dt.timedelta(days=days))
        if action.intervention.is_null():
            return self.nature.sel(time=slice(start, end))
        run = self.runner.run(tag, t0, t0 + dt.timedelta(days=days),
                              restart_in=str(self.root / "restarts" / f"{t0:%Y%m%d}.bin"),
                              restart_out=False, intervention=action.intervention)
        times = self.runner.output_times(run, t0)
        arrays = to_wb2(read_native(run, times), self.static_native)
        ds = standardize(to_dataset(arrays, times), hours=None)
        if "level" in ds.dims:
            ds = ds.sel(level=self.levels)
        head = self.nature.sel(time=slice(start, np.datetime64(t0)))
        timed = [v for v in head.data_vars if "time" in head[v].dims and v in ds]
        merged = xr.concat([head[timed], ds[timed]], dim="time")
        for v in self.static_vars:
            merged[v] = self.nature[v]
        return merged

    def window(self, ds: xr.Dataset, init: dt.datetime, steps: int, action: Action,
               device="cpu") -> Dict[str, torch.Tensor]:
        wds = GraphCastWindowDataset(ds, self.layout, self.stats, steps,
                                     start=np.datetime_as_string(np.datetime64(init)),
                                     end=np.datetime_as_string(np.datetime64(init)))
        if len(wds) != 1:
            raise RuntimeError(f"expected one window at {init}, got {len(wds)}")
        item = wds[0]
        batch = {k: v[None].to(device) for k, v in item.items()}
        batch["action"] = torch.from_numpy(action_field(action.intervention, self.lat, self.lon,
                                                        self.sea))[None].to(device)
        batch["solar_factor"] = torch.tensor([action.intervention.solar_factor], device=device)
        return batch


def rollout_error(model, batch, steps: int, cap: float = 10.0) -> Tuple[torch.Tensor, bool]:
    """GraphCast-weighted error over `steps` (differentiable); blown = non-finite or > cap."""
    attach_loss(model)
    state, ctx = model.begin(batch)
    losses = []
    tmask = torch.zeros_like(batch["targets"][:, 0])
    for k in range(steps):
        state, precip = model.advance(state, ctx, k)
        pred = model.predict(state, ctx, k, precip)
        _, e = errors(model, pred, batch["targets"][:, k], tmask)
        if not (torch.isfinite(e).all() and _finite(state).all()) or float(e.detach()) > cap:
            return (torch.stack(losses).mean() if losses else None), True
        losses.append(e.mean())
    return torch.stack(losses).mean(), False


@dataclasses.dataclass
class GameConfig:
    arm: str = "lp"                 # lp | random | nature (adapt = nature from the ERA5 model) |
                                    # randomized (board 3: random SPEEDY parameters per branch)
    updates: int = 150
    branch_days: int = 6            # SPEEDY branch length per update
    train_steps: int = 8            # 48 h training window starting at day 0 or 1
    probe_day: int = 4              # learning-progress probe window (day 4 -> day 6)
    lr: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    warmup: int = 10
    policy_lr: float = 0.5
    entropy_weight: float = 0.01
    baseline_decay: float = 0.9
    train_start: str = "1982-01-01"
    train_end: str = "1983-12-31"
    seed: int = 0
    save_every: int = 10


class Experimenter:
    """Context-free softmax bandit over the training actions, REINFORCE with an
    EMA baseline on the learning-progress reward."""

    def __init__(self, n: int, cfg: GameConfig):
        self.logits = np.zeros(n)
        self.cfg = cfg
        self.baseline = 0.0

    def probs(self) -> np.ndarray:
        z = self.logits - self.logits.max()
        p = np.exp(z)
        return p / p.sum()

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.choice(len(self.logits), p=self.probs()))

    def update(self, a: int, reward: float) -> None:
        p = self.probs()
        adv = reward - self.baseline
        grad = -p * adv
        grad[a] += adv
        ent_grad = -p * (np.log(p + 1e-12) + (-(p * np.log(p + 1e-12)).sum()))
        self.logits += self.cfg.policy_lr * (grad + self.cfg.entropy_weight * ent_grad)
        self.baseline = self.cfg.baseline_decay * self.baseline + (1 - self.cfg.baseline_decay) * reward

    def entropy(self) -> float:
        p = self.probs()
        return float(-(p * np.log(p + 1e-12)).sum())


# board 3 ranges (log-uniform for the factors that scale a time or a coefficient)
RANDOMIZE = {"conv_factor": (0.5, 2.0), "drag_factor": (0.5, 2.0), "co2_factor": (0.8, 1.25),
             "solar_factor": (0.98, 1.02)}


def random_world(rng: np.random.Generator) -> Action:
    """Board 3: a SPEEDY world with random convection time, surface drag, CO2 and
    solar constant (the model sees solar and CO2 through its action channels;
    convection and drag are hidden and must be inferred from the trajectory)."""
    kw = {}
    for k, (lo, hi) in RANDOMIZE.items():
        kw[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return Action("world " + " ".join(f"{k.split('_')[0]}={v:.2f}" for k, v in kw.items()), Intervention(**kw))


def train_game(model, world: SpeedyWorld, cfg: GameConfig, device="cpu",
               state: Optional[dict] = None, save: Optional[Callable[[dict], None]] = None,
               log: Callable[[str], None] = print) -> dict:
    """Runs `cfg.updates` updates of the world model (and the experimenter for
    arm lp); `state` resumes (history, optimizer, experimenter, rng)."""
    acts = train_actions()
    rng = np.random.default_rng(cfg.seed)
    points = world.branch_points(cfg.train_start, cfg.train_end, cfg.branch_days)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(cfg.warmup, 1))
                                              * 0.5 * (1 + math.cos(math.pi * min(s / cfg.updates, 1.0))))
    exp = Experimenter(len(acts), cfg)
    hist = {"update": [], "action": [], "loss": [], "probe_before": [], "probe_after": [], "reward": [],
            "entropy": [], "blown": [], "seconds": []}
    done = 0
    if state:
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        exp.logits = np.asarray(state["logits"])
        exp.baseline = state["baseline"]
        hist = state["history"]
        rng = np.random.default_rng(state["rng_seed"])
        done = len(hist["update"])
    null = Action("none", Intervention())
    t_start = time.time()
    for n in range(done, cfg.updates):
        t0 = points[int(rng.integers(len(points)))]
        if cfg.arm in ("nature", "adapt"):
            a, act = -1, null
        elif cfg.arm == "randomized":
            a, act = -1, random_world(rng)
        elif cfg.arm == "random":
            a = int(rng.integers(len(acts)))
            act = acts[a]
        else:
            a = exp.sample(rng)
            act = acts[a]
        ds = world.branch(t0, act, cfg.branch_days)
        start_day = int(rng.integers(2))
        batch = world.window(ds, t0 + dt.timedelta(days=start_day), cfg.train_steps, act, device)
        probe = None
        before = after = float("nan")
        if cfg.arm == "lp":
            probe = world.window(ds, t0 + dt.timedelta(days=cfg.probe_day), cfg.train_steps, act, device)
            with torch.no_grad():
                e, blown_p = rollout_error(model, probe, cfg.train_steps)
                before = float(e) if e is not None and not blown_p else 10.0
        model.train()
        opt.zero_grad(set_to_none=True)
        loss, blown = rollout_error(model, batch, cfg.train_steps)
        if loss is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
        sched.step()
        reward = float("nan")
        if cfg.arm == "lp":
            with torch.no_grad():
                e, blown_p = rollout_error(model, probe, cfg.train_steps)
                after = float(e) if e is not None and not blown_p else 10.0
            reward = (before - after) / max(before, 1e-6)
            exp.update(a, reward)
        for k, v in (("update", n), ("action", act.name), ("loss", float(loss) if loss is not None else float("nan")),
                     ("probe_before", before), ("probe_after", after), ("reward", reward),
                     ("entropy", exp.entropy()), ("blown", bool(blown)),
                     ("seconds", time.time() - t_start)):
            hist[k].append(v)
        if (n + 1) % 10 == 0:
            recent = [x for x in hist["loss"][-10:] if x == x]
            log(f"[game {cfg.arm}] {n + 1}/{cfg.updates} loss {np.mean(recent) if recent else float('nan'):.4f} "
                f"entropy {exp.entropy():.3f} blown {sum(hist['blown'][-10:])} ({time.time() - t_start:.0f} s)")
        if save is not None and ((n + 1) % cfg.save_every == 0 or n + 1 == cfg.updates):
            save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "logits": exp.logits.tolist(),
                  "baseline": exp.baseline, "history": hist, "rng_seed": int(rng.integers(2 ** 31))})
    hist["final_probs"] = dict(zip([a.name for a in acts], exp.probs().tolist()))
    return hist


@torch.no_grad()
def response_eval(model, world: SpeedyWorld, points: Sequence[dt.datetime], actions: Sequence[Action],
                  steps: int, device="cpu", log: Callable[[str], None] = print) -> dict:
    """S1/S4: per (branch point, action) the true response (branch - nature) and
    the model's (forecast with the action - forecast without), physical units,
    leads 1..steps. Returns per-pair RMSE of the response error, pattern
    correlation and sign agreement of the late-window (second half) mean
    response, per target channel."""
    model.eval()
    layout = world.layout
    keys = [c.key for c in layout.targets]
    std = model.std.cpu().numpy()[:, None, None]
    lat_w = np.cos(np.deg2rad(world.lat))[:, None]
    null = Action("none", Intervention())
    rows = []
    for t0 in points:
        nat = world.branch(t0, null, int(math.ceil(steps / 4)) + 1)
        b0 = world.window(nat, t0, steps, null, device)
        p0 = model.forward_rollout(b0, steps).cpu().numpy()[0]  # normalised [L, C, H, W]
        truth0 = b0["targets"].cpu().numpy()[0]
        for act in actions:
            ds = world.branch(t0, act, int(math.ceil(steps / 4)) + 1)
            b = world.window(ds, t0, steps, act, device)
            pa = model.forward_rollout(b, steps).cpu().numpy()[0]
            truth = b["targets"].cpu().numpy()[0]
            r_true = (truth - truth0) * std
            r_pred = (pa - p0) * std
            late = slice(steps // 2, steps)
            mt, mp = r_true[late].mean(0), r_pred[late].mean(0)        # [C, H, W]
            row = {"init": t0.isoformat(), "action": act.name, "held_out": act.held_out}
            for j, key in enumerate(keys):
                w = np.broadcast_to(lat_w, mt[j].shape)
                rmse_err = float(np.sqrt((((r_pred[late, j] - r_true[late, j]) ** 2) * w).sum((1, 2)).mean() / w.sum()))
                rmse_zero = float(np.sqrt(((r_true[late, j] ** 2) * w).sum((1, 2)).mean() / w.sum()))
                a, bb = mt[j] - (mt[j] * w).sum() / w.sum(), mp[j] - (mp[j] * w).sum() / w.sum()
                den = math.sqrt(float((a * a * w).sum() * (bb * bb * w).sum()))
                corr = float((a * bb * w).sum() / den) if den > 0 else float("nan")
                big = np.abs(mt[j]) > 0.5 * np.abs(mt[j]).max()
                sign = float((np.sign(mt[j][big]) == np.sign(mp[j][big])).mean()) if big.any() else float("nan")
                row[key] = {"rmse": rmse_err, "rmse_zero": rmse_zero, "corr": corr, "sign": sign}
            rows.append(row)
            log(f"[response] {t0:%Y-%m-%d} {act.name}: t850 corr "
                f"{row.get('temperature@850', {}).get('corr', float('nan')):.2f}")
    return {"rows": rows, "keys": keys}
