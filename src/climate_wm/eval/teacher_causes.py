"""Stage 15.4: score the world model's causal arrows against SPEEDY's (ground truth).

The same cut is made in both worlds from the same initial state:

* SPEEDY (the speedy15 binary): the total physics tendency of T, q or (u, v) is
  scaled to zero (`Intervention.heat_scale / moist_scale / mom_scale`), or the sun
  is switched off (`solar_factor = 0`). Its response is cut run - control run.
* The world model: the matching stage 9 knock-out (eval/physics_causes.py:
  no_heating, no_moistening, no_momentum + no_drag), or, for the sun, the cut run's
  own forcings (TISR = 0 and the solar action), so the sun's path is the learned one.
  Its response is cut forecast - control forecast.

Per cut, block of the equations (physics_causes.groups) and lead:
  corr   area-weighted pattern correlation of the two responses (per-channel centred)
  sign   sign agreement, weighted by |SPEEDY response| (rule T2: >= 0.70)
  ratio  rms model response / rms SPEEDY response (1 = right amplitude)
  gain   1 - err(cut model vs cut truth) / err(uncut model vs cut truth): + when
         applying the cut moves the forecast towards the cut world
"""
from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..sim.intervention_game import Action
from ..sim.speedy import Intervention
from . import physics_causes as pc

# name -> (SPEEDY intervention, matching model knock-outs; empty = the forcings carry the cut)
CUTS: Dict[str, Tuple[Intervention, Tuple[str, ...]]] = {
    "no_heating": (Intervention(heat_scale=0.0), ("no_heating",)),
    "no_moistening": (Intervention(moist_scale=0.0), ("no_moistening",)),
    "no_momentum": (Intervention(mom_scale=0.0), ("no_momentum", "no_drag")),
    "sun_off": (Intervention(solar_factor=0.0), ()),
}
LEADS = (4, 12, 20)  # 24, 72, 120 h


def model_knockout(names: Sequence[str], model) -> Tuple[Optional[pc.Intervention], List[str]]:
    """The physics_causes intervention for the knock-outs `model` has something to cut in."""
    used = [n for n in names if pc.available(n, model)]
    if not used:
        return None, []
    iv = pc.INTERVENTIONS[used[0]]
    for n in used[1:]:
        iv = iv + pc.INTERVENTIONS[n]
    return iv, used


def lat_weights(lat: np.ndarray) -> torch.Tensor:
    w = np.cos(np.deg2rad(np.asarray(lat, np.float64)))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def response_metrics(d_true: torch.Tensor, d_mod: torch.Tensor, chans: Sequence[int],
                     lat_w: torch.Tensor) -> Dict[str, float]:
    """Metrics of the responses [C, H, W] (normalised units) over the channels `chans`."""
    t, m = d_true[list(chans)].double(), d_mod[list(chans)].double()
    nan = {"corr": float("nan"), "sign": float("nan"), "ratio": float("nan"), "rms_true": float("nan")}
    if not torch.isfinite(m).all() or not torch.isfinite(t).all():
        return nan
    w = lat_w.double()[None, :, None].expand_as(t)
    mean = lambda x: (x * w).sum((1, 2), keepdim=True) / w.sum((1, 2), keepdim=True)
    tc, mc = t - mean(t), m - mean(m)
    den = torch.sqrt((w * tc ** 2).sum() * (w * mc ** 2).sum())
    a = w * t.abs()
    tt = (w * t ** 2).sum()
    return {"corr": float((w * tc * mc).sum() / den) if den > 0 else float("nan"),
            "sign": float((a * (torch.sign(m) == torch.sign(t)).double()).sum() / a.sum()) if a.sum() > 0
            else float("nan"),
            "ratio": float(torch.sqrt((w * m ** 2).sum() / tt)) if tt > 0 else float("nan"),
            "rms_true": float(torch.sqrt(tt / w.sum()))}


def _mse(a: torch.Tensor, b: torch.Tensor, chans: Sequence[int], lat_w: torch.Tensor) -> float:
    d = (a[list(chans)] - b[list(chans)]).double() ** 2
    w = lat_w.double()[None, :, None].expand_as(d)
    return float((d * w).sum() / w.sum())


def score_init(base, models: Dict[str, object], world, t0: dt.datetime, steps: int, days: int,
               groups: Dict[str, List[int]], lat_w: torch.Tensor, device="cpu",
               cuts: Dict[str, Tuple[Intervention, Tuple[str, ...]]] = CUTS) -> dict:
    """One initialisation: SPEEDY control and cut branches, model control and cut forecasts."""
    null = Action("control", Intervention())
    ctrl = world.window(world.branch(t0, null, days), t0, steps, null, device)
    p0 = pc.rollout(base, ctrl, steps)[0]
    truth0 = ctrl["targets"][0].float()
    row = {"init": t0.isoformat(), "cuts": {}}
    for name, (iv_s, _) in cuts.items():
        act = Action(name, iv_s)
        b = world.window(world.branch(t0, act, days, tag=f"cut_{name}"), t0, steps, act, device)
        p1 = pc.rollout(models[name], b, steps)[0]
        truth1 = b["targets"][0].float()
        out = {}
        for k in LEADS:
            if k > steps:
                continue
            j = k - 1
            for g, chans in groups.items():
                m = response_metrics(truth1[j] - truth0[j], p1[j] - p0[j], chans, lat_w)
                e_cut, e_blind = _mse(p1[j], truth1[j], chans, lat_w), _mse(p0[j], truth1[j], chans, lat_w)
                m["gain"] = (1.0 - e_cut / e_blind) if np.isfinite(e_cut) and e_blind > 0 else float("nan")
                m["model_err"] = _mse(p0[j], truth0[j], chans, lat_w)
                out[f"{g}@{k}"] = m
        out["blown"] = bool(not torch.isfinite(p1).all())
        row["cuts"][name] = out
    row["blown_control"] = bool(not torch.isfinite(p0).all())
    return row


def summarize(rows: Sequence[dict], groups: Sequence[str], cuts: Sequence[str]) -> dict:
    """Means over the initialisations (blown runs left out) and the rule T2 numbers."""
    out = {"inits": len(rows), "cuts": {}}
    signs72 = []
    for c in cuts:
        res = {"blown": int(sum(r["cuts"][c]["blown"] for r in rows))}
        for k in LEADS:
            for g in groups:
                key = f"{g}@{k}"
                vals = [r["cuts"][c][key] for r in rows if key in r["cuts"][c]]
                if not vals:
                    continue
                res[key] = {m: float(np.nanmean([v[m] for v in vals])) if np.isfinite([v[m] for v in vals]).any()
                            else float("nan") for m in vals[0]}
                if k == 12:
                    signs72.append(res[key]["sign"])
        out["cuts"][c] = res
    out["T2_sign_72h"] = float(np.nanmean(signs72)) if signs72 else float("nan")
    return out


def evaluate_teacher_causes(model_cfg: dict, state: dict, layout, lat, lon, world, points: Sequence[dt.datetime],
                            steps: int = 20, days: int = 5, device="cpu",
                            cuts: Dict[str, Tuple[Intervention, Tuple[str, ...]]] = CUTS,
                            done: Optional[List[dict]] = None, save: Optional[Callable[[List[dict]], None]] = None,
                            log: Callable[[str], None] = print) -> dict:
    base = pc.build(model_cfg, state, layout, lat, lon, None, device)
    models, used = {}, {}
    for name, (_, names) in cuts.items():
        iv, used[name] = model_knockout(names, base)
        models[name] = pc.build(model_cfg, state, layout, lat, lon, iv, device) if iv is not None else base
        if names and not used[name]:
            raise ValueError(f"{name}: the model has none of {names} to cut")
    groups = pc.groups(layout)
    lat_w = lat_weights(lat).to(device)
    rows = list(done or [])
    for t0 in points[len(rows):]:
        rows.append(score_init(base, models, world, t0, steps, days, groups, lat_w, device, cuts))
        r = rows[-1]["cuts"]
        log(f"[causes15] {t0:%Y-%m-%d} " + " ".join(
            f"{c} T sign {r[c].get('thermodynamics@12', {}).get('sign', float('nan')):.2f}" for c in cuts))
        if save is not None:
            save(rows)
    res = summarize(rows, list(groups), list(cuts))
    res["model_knockouts"] = used
    res["rows"] = rows
    return res
