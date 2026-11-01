"""Stage 15.3 (simulator as teacher): train the physics world model on SPEEDY
episodes with the state loss plus a loss on the column's physics tendencies.

    L = L_state (GraphCast-weighted MSE over the rollout)
      + lambda_tend * L_tendency (normalised, latitude-weighted MSE of the mean column
        forcing of each 6 h step against SPEEDY's per-process sum over the same 6 h)

N1-S = lambda_tend 0, N1-T = lambda_tend > 0. The rollout grows by a curriculum
(1 step -> 24 h -> 48 h). Episodes come from sim/episodes.py; the split is by
episode. Group C episodes carry their CO2 and solar factors as the stage 14
action channels; their convection and drag factors stay hidden.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import xarray as xr

from ..data.era5_lazy_loader import GraphCastWindowDataset, standardize
from ..sim.episodes import episode_tendencies
from ..sim.intervention_game import action_field
from ..sim.speedy import Intervention
from .physics_game import _finite, attach_loss
from .prediction_game import errors

TEND_KEYS = ("Q", "S", "F_u", "F_v")   # the order of PhysicsWorldModel.last_tendency


@dataclasses.dataclass
class TeacherConfig:
    curriculum: Tuple[Tuple[int, int], ...] = ((1, 150), (4, 150), (8, 100))  # (rollout steps, updates)
    lambda_tend: float = 1.0
    tend_keys: Tuple[str, ...] = TEND_KEYS     # supervised parts (ablation: ("Q",), ("Q", "S"))
    groups: Tuple[str, ...] = ("A", "B", "C")
    init_stride: int = 2                       # window starts every 12 h
    lr: float = 1e-4
    grad_clip: float = 1.0
    cap: float = 10.0
    seed: int = 0
    save_every: int = 25
    log_every: int = 10

    @classmethod
    def from_dict(cls, d: dict) -> "TeacherConfig":
        d = dict(d)
        if "curriculum" in d:
            d["curriculum"] = tuple(tuple(int(v) for v in c) for c in d["curriculum"])
        for k in ("tend_keys", "groups"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)


class EpisodeSet:
    """The episodes of one split as window datasets, with tendency targets and actions."""

    def __init__(self, root: str, layout, stats, split: str = "train", groups: Sequence[str] = ("A", "B", "C")):
        self.root = Path(root)
        self.layout, self.stats = layout, stats
        specs = json.loads((self.root / "episodes.json").read_text())
        self.specs, self.ds = [], []
        for s in specs:
            path = self.root / f"{s['name']}.zarr"
            if s["split"] != split or s["group"] not in groups or not (path / "zarr.json").exists():
                continue
            ds = xr.open_zarr(path)
            if not ds.attrs.get("complete", False):
                continue
            self.specs.append(s)
            self.ds.append(standardize(ds, hours=None))
        if not self.ds:
            raise ValueError(f"no complete {split} episodes of groups {groups} in {root}")
        self.levels = [int(l) for l in layout.task.pressure_levels]
        self._wins: Dict[int, List[GraphCastWindowDataset]] = {}
        self._std: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.ds)

    def datasets(self, steps: int) -> List[GraphCastWindowDataset]:
        if steps not in self._wins:
            out = []
            for s, ds in zip(self.specs, self.ds):
                first = np.datetime64(dt.datetime.fromisoformat(s["start"]) +
                                      dt.timedelta(days=s.get("spinup_days", 0), hours=6))
                out.append(GraphCastWindowDataset(ds, self.layout, self.stats, steps,
                                                  start=np.datetime_as_string(first + np.timedelta64(6, "h"))))
            self._wins[steps] = out
        return self._wins[steps]

    def windows(self, steps: int, stride: int = 1) -> List[Tuple[int, int]]:
        return [(e, j) for e, w in enumerate(self.datasets(steps)) for j in range(0, len(w), stride)]

    def tendency(self, e: int, times: Sequence[np.datetime64]) -> np.ndarray:
        """[T, 4 L, lat, lon] float32 in physical units (Q K/s, S kg/kg/s, F m/s^2)."""
        t = episode_tendencies(self.ds[e].sel(time=list(times)))
        parts = [t[k].sel(level=self.levels).transpose("time", "level", "lat", "lon").values
                 for k in TEND_KEYS]
        return np.concatenate(parts, 1).astype(np.float32)

    def tendency_std(self, n: int = 24, seed: int = 0, floor: float = 0.05) -> np.ndarray:
        """Per channel (key, level) std of the targets [4 L] from n random frames, floored at
        `floor` x the largest std of the same part: SPEEDY has no friction above the boundary
        layer and almost no moistening at 50 hPa, and a near-zero std there would blow the
        normalised error of any non-zero model tendency up by 1e6 or more."""
        if self._std is None:
            rng = np.random.default_rng(seed)
            frames = []
            for _ in range(n):
                e = int(rng.integers(len(self.ds)))
                times = self.ds[e].time.values
                frames.append(self.tendency(e, [times[int(rng.integers(2, len(times)))]]))
            x = np.concatenate(frames, 0)
            std = x.std((0, 2, 3)).reshape(len(TEND_KEYS), -1)
            std = np.maximum(std, floor * std.max(1, keepdims=True))
            self._std = np.maximum(std.reshape(-1), 1e-12).astype(np.float32)
        return self._std

    def batch(self, e: int, j: int, steps: int, device="cpu") -> Dict[str, torch.Tensor]:
        item = self.datasets(steps)[e][j]
        batch = {k: v[None].to(device) for k, v in item.items()}
        t0 = np.datetime64(int(batch["init_time"][0]), "s")
        times = [t0 + np.timedelta64(6 * (k + 1), "h") for k in range(steps)]
        batch["tendency"] = torch.from_numpy(self.tendency(e, times))[None].to(device)
        iv = Intervention(**{k: v for k, v in self.specs[e]["intervention"].items() if k != "patches"})
        ds = self.ds[e]
        sea = 1.0 - ds["land_sea_mask"].transpose("lat", "lon").values
        batch["action"] = torch.from_numpy(action_field(iv, ds.lat.values, ds.lon.values,
                                                        sea))[None].to(device)
        batch["solar_factor"] = torch.tensor([iv.solar_factor], device=device)
        return batch


def to_data_grid(model, x: torch.Tensor) -> torch.Tensor:
    sh = model.core.sh
    return sh.to_grid(sh.to_spec(x), data=True)


def tendency_error(model, pred_g: torch.Tensor, target: torch.Tensor, std: torch.Tensor,
                   keys: Sequence[str] = TEND_KEYS) -> torch.Tensor:
    """Latitude-weighted MSE of the normalised column forcing on the data grid over the
    supervised parts, points below the surface (target exactly 0 in every part) left out."""
    n = target.shape[1] // len(TEND_KEYS)
    pred = to_data_grid(model, pred_g)
    w = model.loss_fn.lat_w[None, None, :, None]
    alive = (target.abs().view(target.shape[0], len(TEND_KEYS), n, *target.shape[-2:]).sum(1) > 0).float()
    sel = torch.cat([torch.arange(i * n, (i + 1) * n) for i, k in enumerate(TEND_KEYS) if k in keys])
    err = ((pred[:, sel] - target[:, sel]) / std[sel, None, None]).square()
    m = alive.repeat(1, len(sel) // n, 1, 1) * w
    return (err * m).sum() / m.expand_as(err).sum().clamp_min(1e-12)


def teacher_rollout(model, batch, steps: int, std: torch.Tensor, cfg: TeacherConfig):
    """(state loss, tendency loss, blown) over the window (differentiable)."""
    attach_loss(model)
    model.track_tendency = True
    try:
        state, ctx = model.begin(batch)
        ls, lt = [], []
        tmask = torch.zeros_like(batch["targets"][:, 0])
        for k in range(steps):
            state, precip = model.advance(state, ctx, k)
            pred = model.predict(state, ctx, k, precip)
            _, e = errors(model, pred, batch["targets"][:, k], tmask)
            te = (tendency_error(model, model.last_tendency, batch["tendency"][:, k], std, cfg.tend_keys)
                  if model.last_tendency is not None else e.new_zeros(()))  # F1: no column
            if not (torch.isfinite(e).all() and torch.isfinite(te) and _finite(state).all()) \
                    or float(e.detach()) > cfg.cap:
                return ((torch.stack(ls).mean() if ls else None), (torch.stack(lt).mean() if lt else None), True)
            ls.append(e.mean())
            lt.append(te)
        return torch.stack(ls).mean(), torch.stack(lt).mean(), False
    finally:
        model.track_tendency = False
        model.last_tendency = None


def train_teacher(model, data, cfg: TeacherConfig, device="cpu", progress: Optional[dict] = None,
                  save: Optional[Callable[[dict], None]] = None, log: Callable[[str], None] = print) -> dict:
    """Curriculum training; `progress` resumes (history, optimizer, rng, update)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    std = torch.from_numpy(data.tendency_std(seed=cfg.seed)).to(device)
    hist = {"update": [], "steps": [], "state": [], "tend": [], "blown": []}
    start = 0
    if progress:
        hist, start = progress["history"], progress["update"]
        opt.load_state_dict(progress["opt"])
        rng.bit_generator.state = progress["rng"]
    u = 0
    for steps, n in cfg.curriculum:
        wins = data.windows(steps, cfg.init_stride)
        for _ in range(n):
            if u < start:
                u += 1
                continue
            e, j = wins[int(rng.integers(len(wins)))]
            batch = data.batch(e, j, steps, device)
            model.train()
            opt.zero_grad(set_to_none=True)
            ls, lt, blown = teacher_rollout(model, batch, steps, std, cfg)
            if ls is not None:
                loss = ls + cfg.lambda_tend * lt
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                if all(p.grad is None or torch.isfinite(p.grad).all() for p in params):
                    opt.step()
            hist["update"].append(u)
            hist["steps"].append(steps)
            hist["state"].append(float(ls.detach()) if ls is not None else float("nan"))
            hist["tend"].append(float(lt.detach()) if lt is not None else float("nan"))
            hist["blown"].append(bool(blown))
            u += 1
            if u % cfg.log_every == 0:
                k = cfg.log_every
                log(f"[teacher] {u} steps {steps} state {np.nanmean(hist['state'][-k:]):.4f} "
                    f"tend {np.nanmean(hist['tend'][-k:]):.4f} blown {sum(hist['blown'][-k:])}")
            if save is not None and u % cfg.save_every == 0:
                save({"history": hist, "update": u, "opt": opt.state_dict(), "rng": rng.bit_generator.state})
    hist["tend_std"] = std.cpu().tolist()
    return hist


@torch.no_grad()
def evaluate_teacher(model, data, steps: int, stride: int, std: torch.Tensor, device="cpu",
                     log: Callable[[str], None] = print) -> dict:
    """Per test window: state error per lead (GraphCast-weighted, normalised), tendency
    error and pattern correlation of the column forcing with SPEEDY's (T1) per part."""
    attach_loss(model)
    model.eval()
    n_lev = None
    rows = []
    for e, j in data.windows(steps, stride):
        batch = data.batch(e, j, steps, device)
        model.track_tendency = True
        state, ctx = model.begin(batch)
        tmask = torch.zeros_like(batch["targets"][:, 0])
        row = {"episode": data.specs[e]["name"], "group": data.specs[e]["group"], "state": [], "corr": []}
        for k in range(steps):
            state, precip = model.advance(state, ctx, k)
            pred = model.predict(state, ctx, k, precip)
            _, err = errors(model, pred, batch["targets"][:, k], tmask)
            row["state"].append(float(err))
            t = batch["tendency"][0, k]
            n_lev = t.shape[0] // len(TEND_KEYS)
            cors = []
            if model.last_tendency is None:  # F1 (no column): no tendency to score
                row["corr"].append([float("nan")] * len(TEND_KEYS))
                continue
            p = to_data_grid(model, model.last_tendency)[0]
            for i in range(len(TEND_KEYS)):
                a, b = p[i * n_lev:(i + 1) * n_lev].flatten(), t[i * n_lev:(i + 1) * n_lev].flatten()
                a, b = a - a.mean(), b - b.mean()
                cors.append(float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)))
            row["corr"].append(cors)
        model.track_tendency = False
        rows.append(row)
    st = np.array([r["state"] for r in rows])
    cr = np.array([r["corr"] for r in rows])
    out = {"steps": steps, "windows": len(rows), "state_mean": np.nanmean(st, 0).tolist(),
           "corr_mean": {k: np.nanmean(cr[:, :, i], 0).tolist() for i, k in enumerate(TEND_KEYS)},
           "blown": int((~np.isfinite(st)).any(1).sum()), "rows": rows}
    log(f"[teacher eval] {len(rows)} windows, state {out['state_mean'][0]:.4f} .. {out['state_mean'][-1]:.4f}, "
        f"Q corr {out['corr_mean']['Q'][0]:.3f}, S corr {out['corr_mean']['S'][0]:.3f}")
    return out
