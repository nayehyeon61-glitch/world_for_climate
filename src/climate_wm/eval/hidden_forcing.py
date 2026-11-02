"""Stage 15.5: does the memory infer a hidden forcing? (README_KO section 15, rules T4-T6.)

A SPEEDY branch carries a persistent SST anomaly (a few patches of +-2 K) that the
world model never sees: its inputs are the atmospheric state only. The memory
model re-observes the branch for `h` days (the fill: one-step forecasts, the
posterior absorbs each innovation, as in compare_memory.ContinuousFill), then
runs free for `free_steps` six-hour steps. The same is done

  * with the memory frozen (`frozen`: the filled nature memory m0 of the start date kept fixed, no
    recurrence and no innovation, so the branch's history never enters) and held at zero (`markov`:
    the same column with m = 0). The "memory" knock-out is not used: it also drops the tendency
    network's m-independent part, which the SPEEDY-adapted models rely on (they blow up at once),
  * on the nature run from the same date (no anomaly): the memory's gain that has
    nothing to do with the hidden forcing (it also corrects the column's drift),
  * optionally with the anomaly shown through the action channel (`shown`).

The memory's gain = 1 - error(memory) / error(frozen). Hidden-forcing memory gain = that gain on
the anomaly branch minus the gain on the nature run (a difference in differences), at each history length h; rule T4 asks for
>= 5% at 5 days with the forcing hidden and < 2% with it shown, T6 for a gain that
grows with h and saturates. `probe` (T5) regresses the anomaly on the filled
memory state, column by column, holding out whole branch points.

Errors: the GraphCast-weighted all-variable MSE (normalised units), and t850 /
t2m RMSE (K) weighted by the anomaly's footprint (|SST anomaly| / amplitude x
cos lat), where the forced signal is (SNR 2-5 near a patch at 48 h; 14.2).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..sim.intervention_game import SITES, Action, action_field
from ..sim.speedy import Intervention
from ..train.physics_game import attach_loss
from ..train.prediction_game import errors

STEPS_PER_DAY = 4


@dataclasses.dataclass
class HiddenConfig:
    histories_days: Tuple[int, ...] = (0, 1, 5, 10, 30)
    free_steps: int = 20            # 5 days
    n_patches: int = 3
    amp: float = 2.0                # K
    radius: float = 12.0            # e-folding radius, degrees
    cap: float = 10.0               # a step whose all-variable error exceeds this counts as a blow-up
    seed: int = 0


def hidden_anomaly(rng: np.random.Generator, cfg: HiddenConfig) -> Intervention:
    """`n_patches` distinct SST sites with random signs."""
    idx = rng.choice(len(SITES), cfg.n_patches, replace=False)
    signs = rng.choice([-1.0, 1.0], cfg.n_patches)
    return Intervention(patches=tuple((float(SITES[i][0]), float(SITES[i][1]), cfg.radius, float(s * cfg.amp))
                                      for i, s in zip(idx, signs)))


@torch.no_grad()
def fill_then_free(model, batch, fill_steps: int, free_steps: int, mem0: Optional[torch.Tensor] = None,
                   knock: Iterable[str] = (), cap: float = 10.0):
    """Re-observe `fill_steps` steps, then run `free_steps` free. Returns the free forecasts
    (normalised [free, C, H, W] numpy, NaN after a blow-up), the memory after the fill (or None)
    and whether the run blew up."""
    attach_loss(model)
    mem_k = getattr(model, "memory", None)
    if mem_k is not None:
        mem_k.knock = set(knock)
    t2d = model.layout.target_to_dynamic
    try:
        state, ctx = model.begin(batch)
        snap = batch["inputs"][:, -1]
        mask = torch.zeros_like(snap)
        if mem_k is not None:
            if mem0 is not None:
                state["mem"] = mem0.to(snap.device)
            state = model.posterior(state, ctx, snap, snap, mask)
        mem_after = state["mem"].detach().clone() if (mem_k is not None and fill_steps == 0) else None
        preds, blown = [], False
        tmask = torch.zeros_like(batch["targets"][:, 0])
        for k in range(fill_steps + free_steps):
            state, precip = model.advance(state, ctx, k)
            pred = model.predict(state, ctx, k, precip)
            if k < fill_steps:
                truth = batch["targets"][:, k][:, t2d]
                state = model.posterior(state, ctx, torch.nan_to_num(pred[:, t2d]), truth, mask)
                if mem_k is not None and not torch.isfinite(state["mem"]).all():
                    state["mem"] = torch.zeros_like(state["mem"])
                if k == fill_steps - 1 and mem_k is not None:
                    mem_after = state["mem"].detach().clone()
                continue
            _, e = errors(model, pred, batch["targets"][:, k], tmask)
            if not torch.isfinite(pred).all() or float(e.mean()) > cap:
                blown = True
                break
            preds.append(pred[0].cpu().numpy())
    finally:
        if mem_k is not None:
            mem_k.knock = set()
    out = np.full((free_steps,) + tuple(batch["targets"].shape[2:]), np.nan, np.float32)
    if preds:
        out[:len(preds)] = np.stack(preds)
    return out, mem_after, blown


def footprint(iv: Intervention, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """|SST anomaly| / amplitude on a grid (sea everywhere): the weights of the local errors."""
    f = np.abs(action_field(iv, lat, lon, np.ones((len(lat), len(lon))))[0])
    return f / max(float(f.max()), 1e-9)


def scores(model, pred: np.ndarray, targets: np.ndarray, lat: np.ndarray, foot: np.ndarray,
           keys: Sequence[str]) -> Dict[str, List[float]]:
    """Per free lead: all-variable weighted MSE (normalised) and footprint-weighted t850 / t2m RMSE (K)."""
    attach_loss(model)
    p = torch.from_numpy(np.nan_to_num(pred, nan=0.0))[:, None]
    t = torch.from_numpy(targets)[:, None]
    tmask = torch.zeros_like(t[0])
    alls = []
    for k in range(pred.shape[0]):
        if not np.isfinite(pred[k]).all():
            alls.append(float("nan"))
            continue
        _, e = errors(model, p[k], t[k], tmask)
        alls.append(float(e.mean()))
    std = model.std.cpu().numpy()
    w = np.cos(np.deg2rad(lat))[:, None] * foot
    out = {"all": alls}
    for name, key in (("t850", "temperature@850"), ("t2m", "2m_temperature")):
        if key not in keys:
            continue
        j = list(keys).index(key)
        d2 = ((pred[:, j] - targets[:, j]) * std[j]) ** 2
        out[name] = [float(np.sqrt((d2[k] * w).sum() / w.sum())) for k in range(pred.shape[0])]
    return out


@torch.no_grad()
def hidden_forcing_eval(models: Dict[str, object], world, points: Sequence[dt.datetime], cfg: HiddenConfig,
                        snaps: Optional[Dict[int, torch.Tensor]] = None, shown: Iterable[str] = (),
                        device="cpu", log: Callable[[str], None] = print, done: Optional[dict] = None,
                        save: Optional[Callable[[dict], None]] = None) -> dict:
    """Rows per (point, truth, model, variant, h): free-run scores, blow-up; and the filled memory
    with the anomaly footprint on the core grid for the probe. `shown` labels also run with the
    anomaly in the action channel (on the anomaly branch). Resumable per (point, truth) via `done`."""
    res = done or {"rows": [], "probe": [], "points": []}
    rng = np.random.default_rng(cfg.seed)
    hmax = max(cfg.histories_days)
    steps = STEPS_PER_DAY * hmax + cfg.free_steps
    days = int(math.ceil(steps / STEPS_PER_DAY)) + 1
    null = Action("none", Intervention())
    keys = [c.key for c in world.layout.targets]
    lat, lon = np.asarray(world.lat), np.asarray(world.lon)
    shown = set(shown)
    anomalies = [hidden_anomaly(rng, cfg) for _ in points]
    truths_done = res.setdefault("truths", [])          # finer resume: (point, truth) pairs already scored
    for i, t0 in enumerate(points):
        if t0.isoformat() in res["points"]:
            continue
        iv = anomalies[i]
        foot = footprint(iv, lat, lon)
        hidden_act = Action("hidden_sst", iv)
        branches = {"anomaly": world.branch(t0, hidden_act, days, tag="hidden"), "nature": world.branch(t0, null, days)}
        mem0 = None
        for truth, ds in branches.items():
            if f"{t0.isoformat()}|{truth}" in truths_done:
                continue
            hid = world.window(ds, t0, steps, null, device)
            if snaps is not None:
                mem0 = snaps.get(int(hid["init_time"][0]))   # the filled memory at t0, as begin_filled
            target_np = hid["targets"].cpu().numpy()[0]
            runs = [(lab, False, hid) for lab in models]
            if truth == "anomaly":
                runs += [(lab, True, world.window(ds, t0, steps, hidden_act, device)) for lab in models if lab in shown]
            for lab, is_shown, batch in runs:
                model = models[lab]
                variants = ("memory",)
                if getattr(model, "memory", None) is not None:
                    variants = ("memory", "frozen") if is_shown else ("memory", "frozen", "markov")
                for variant in variants:
                    knock = ("frozen",) if variant in ("frozen", "markov") else ()
                    start = None if variant == "markov" else mem0      # markov: m = 0 from begin, kept there
                    for h in cfg.histories_days:
                        fill = STEPS_PER_DAY * h
                        pred, mem, blown = fill_then_free(model, batch, fill, cfg.free_steps, start, knock, cfg.cap)
                        tg = target_np[fill:fill + cfg.free_steps]
                        row = {"init": t0.isoformat(), "truth": truth, "model": lab, "shown": is_shown,
                               "variant": variant, "h": h, "blown": blown,
                               "patches": [list(p) for p in iv.patches], **scores(model, pred, tg, lat, foot, keys)}
                        res["rows"].append(row)
                        if variant == "memory" and mem is not None and not is_shown:
                            g = model._to_gauss(torch.from_numpy(
                                action_field(iv if truth == "anomaly" else Intervention(), lat, lon,
                                             np.ones((len(lat), len(lon)))))[None].to(mem.device))[0, 0]
                            res["probe"].append({"init": t0.isoformat(), "truth": truth, "model": lab, "h": h,
                                                 "mem": mem[0].cpu().numpy().astype(np.float32),
                                                 "sst": g.cpu().numpy().astype(np.float32)})
                        a = row["all"][-1]
                        log(f"[hidden] {t0:%Y-%m-%d} {truth} {lab}{' shown' if is_shown else ''} {variant} h={h}d: "
                            f"all 120 h {a:.3f}, t850 near patch {row.get('t850', [np.nan])[-1]:.2f} K"
                            f"{' (blown)' if blown else ''}")
            truths_done.append(f"{t0.isoformat()}|{truth}")
            if save is not None:
                save(res)
        res["points"].append(t0.isoformat())
        if save is not None:
            save(res)
    return res


@torch.no_grad()
def forced_signal(model, world, points: Sequence[dt.datetime], cfg: HiddenConfig, device="cpu",
                  tag: str = "hsignal") -> List[dict]:
    """How big the hidden forcing is in the truth: anomaly branch minus nature run, per history h and free
    lead, in the units of the rows (all-variable weighted MSE, normalised; footprint-weighted t850 / t2m
    RMS, K). The anomalies are those of `hidden_forcing_eval` (same seed and points). A model that knows
    nothing of the anomaly has error^2 ~ e_nature^2 + signal^2 on the anomaly branch; knowing it perfectly
    removes the signal^2, which bounds the memory's possible gain (the ceiling in report_hidden_forcing)."""
    rng = np.random.default_rng(cfg.seed)
    steps = STEPS_PER_DAY * max(cfg.histories_days) + cfg.free_steps
    days = int(math.ceil(steps / STEPS_PER_DAY)) + 1
    null = Action("none", Intervention())
    keys = [c.key for c in world.layout.targets]
    lat, lon = np.asarray(world.lat), np.asarray(world.lon)
    anomalies = [hidden_anomaly(rng, cfg) for _ in points]
    out = []
    for iv, t0 in zip(anomalies, points):
        foot = footprint(iv, lat, lon)
        a = world.window(world.branch(t0, Action("hidden_sst", iv), days, tag=tag), t0, steps, null, device)
        n = world.window(world.branch(t0, null, days), t0, steps, null, device)
        a, n = a["targets"].cpu().numpy()[0], n["targets"].cpu().numpy()[0]
        for h in cfg.histories_days:
            fill = STEPS_PER_DAY * h
            out.append({"init": t0.isoformat(), "h": h, "patches": [list(p) for p in iv.patches],
                        **scores(model, a[fill:fill + cfg.free_steps], n[fill:fill + cfg.free_steps], lat, foot, keys)})
    return out


def probe_r2(records: List[dict], ridge: float = 1.0) -> Dict[Tuple[str, int], float]:
    """T5: linear map from the memory channels of a column to the SST anomaly under it, fitted on all
    branch points but one and scored on the held-out one (anomaly branches; the nature runs, with a
    zero anomaly, are in the fit as negatives). R^2 over the held-out anomaly columns, per (model, h)."""
    out = {}
    for key in sorted({(r["model"], r["h"]) for r in records}):
        rs = [r for r in records if (r["model"], r["h"]) == key]
        inits = sorted({r["init"] for r in rs})
        if len(inits) < 2:
            continue
        sse, sst = 0.0, 0.0
        for held in inits:
            fit = [r for r in rs if r["init"] != held]
            X = np.concatenate([r["mem"].reshape(r["mem"].shape[0], -1).T for r in fit])
            y = np.concatenate([r["sst"].ravel() for r in fit])
            Xb = np.hstack([X, np.ones((len(X), 1))])
            w = np.linalg.solve(Xb.T @ Xb + ridge * np.eye(Xb.shape[1]), Xb.T @ y)
            for r in rs:
                if r["init"] == held and r["truth"] == "anomaly":
                    Xt = np.hstack([r["mem"].reshape(r["mem"].shape[0], -1).T, np.ones((r["sst"].size, 1))])
                    yt = r["sst"].ravel()
                    sse += float(((Xt @ w - yt) ** 2).sum())
                    sst += float(((yt - yt.mean()) ** 2).sum())
        out[key] = 1 - sse / sst if sst > 0 else float("nan")
    return out
