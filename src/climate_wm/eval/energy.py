"""Stage 13: blow-ups and the column's energy input (감자 10:28 KST 11/01).

The learned column physics has no energy budget: the heating Q can create
energy, and the blow-ups of stages 9 and 10 all involve the column (N0 without
it is stable). E1 bounds the column's energy input at run time
(PhysicsWorldModel.energy_bound): the moist static energy flux of every column
is clamped to +-`energy_max_flux` W/m^2 and, with `energy_no_work`, the
momentum forcing cannot do work on the wind. Both apply to existing
checkpoints without retraining.

`diagnose_energy` runs free forecasts (or forecasts from a masked snapshot)
under several settings and records, per 6 h step, the all-point error and the
energy terms the model tracks (global means of the heating, moistening and
work fluxes; the largest flux and work of any column; the clamped fraction),
so the term that grows before a blow-up can be read off (rule B4) and the
cost of the bound on a stable model measured (B1).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..train.dreamer_game import play_memory
from ..train.prediction_game import sample_mask

SETTINGS: Dict[str, dict] = {
    "none": {},
    "flux1000": {"energy_max_flux": 1000.0},
    "flux500": {"energy_max_flux": 500.0},
    "flux250": {"energy_max_flux": 250.0},
    "nowork": {"energy_no_work": True},
    "flux500_nowork": {"energy_max_flux": 500.0, "energy_no_work": True},
}
TERMS = ("heat_mean", "moist_mean", "work_mean", "flux_max", "work_max", "clamped")


def set_energy(model, setting: dict) -> None:
    model.cfg.column.energy_max_flux = setting.get("energy_max_flux")
    model.cfg.column.energy_no_work = bool(setting.get("energy_no_work", False))


@torch.no_grad()
def diagnose_energy(model, dataset, indices: Sequence[int], steps: int, settings: Dict[str, dict] = SETTINGS,
                    kinds: Sequence[str] = ("none",), region_fraction: float = 0.25, seed: int = 0,
                    cap: float = 10.0, device="cpu", log=None, progress_path: Optional[str] = None) -> Dict[str, object]:
    """Per kind ('none' = the full snapshot, 'region' = a hidden cap filled with the
    mean, as the stage 7-10 game) and setting: the all-point errors per lead
    (capped; a blown-up run scores `cap` from its blow-up on), the blow-up step,
    and the energy terms per lead. Resumable."""
    model.eval()
    done: Dict[str, List[dict]] = {}
    if progress_path is not None and os.path.exists(progress_path):
        done = torch.load(progress_path, weights_only=False)
    recs = done.setdefault("recs", [])
    try:
        for n, i in enumerate(indices):
            if n < len(recs):
                continue
            item = dataset[int(i)]
            batch = {k: v[None].to(device) for k, v in item.items()}
            rec: Dict[str, object] = {"init": int(item["init_time"])}
            for kind in kinds:
                gen = torch.Generator().manual_seed(seed + 1000 * n)
                if kind == "none":
                    mask = torch.zeros(len(model.layout.dynamic), len(dataset.lat), len(dataset.lon))
                else:
                    mask, _ = sample_mask(model.layout, kind, dataset.lat, dataset.lon, gen, region_fraction)
                for name, setting in settings.items():
                    set_energy(model, setting)
                    model.energy_track = []
                    _, a = play_memory(model, None, batch, mask[None], steps, obs_prob=0.0, rule="posterior", cap=cap)
                    track = torch.stack(model.energy_track)[:, 0]  # [calls, 6]
                    per = max(1, track.shape[0] // steps)
                    energy = track[:per * steps].view(steps, per, -1).mean(1)
                    err = a[0].cpu()
                    blown = torch.nonzero(err >= cap)
                    rec[f"{kind}:{name}"] = {"err": err.tolist(), "energy": energy.tolist(),
                                             "blowup_step": int(blown[0]) if len(blown) else None}
            recs.append(rec)
            if progress_path is not None:
                torch.save(done, progress_path + ".tmp")
                os.replace(progress_path + ".tmp", progress_path)
            if log is not None:
                log(f"[energy] {n + 1}/{len(indices)}: " + ", ".join(
                    f"{k} {'blown@' + str(v['blowup_step']) if v['blowup_step'] is not None else 'ok'}"
                    for k, v in rec.items() if k != "init"))
    finally:
        set_energy(model, {})
        model.energy_track = None
    out: Dict[str, object] = {"lead_h": [6 * (k + 1) for k in range(steps)], "terms": list(TERMS),
                              "settings": {k: v for k, v in settings.items()}, "kinds": list(kinds),
                              "inits": [r["init"] for r in recs], "runs": {}}
    for kind in kinds:
        base_rows = [r[f"{kind}:none"] for r in recs] if "none" in settings else None
        for name in settings:
            rows = [r[f"{kind}:{name}"] for r in recs]
            err = np.array([r["err"] for r in rows])
            blown = [r["blowup_step"] for r in rows]
            run = {"blowups": int(sum(b is not None for b in blown)), "blowup_steps": blown,
                   "err": err.mean(0).tolist(), "err_rows": err.tolist(),  # per initialisation (arm comparisons)
                   "energy": np.mean([r["energy"] for r in rows], 0).tolist()}
            if base_rows is not None and name != "none":
                ok = np.array([b is None and r["blowup_step"] is None for b, r in zip(blown, base_rows)])
                if ok.any():
                    base = np.array([r["err"] for r in base_rows])[ok]
                    run["err_ratio_finite"] = (err[ok].mean(0) / base.mean(0)).tolist()
                    run["finite_both"] = int(ok.sum())
            # the energy terms of the blown-up runs over the four steps before the blow-up,
            # against the same leads of the runs that stayed finite
            before, same = [], []
            for r in rows:
                b = r["blowup_step"]
                if b is not None and b >= 2:
                    before.append(np.array(r["energy"][max(0, b - 4):b]).mean(0))
                    for q in rows:
                        if q["blowup_step"] is None:
                            same.append(np.array(q["energy"][max(0, b - 4):b]).mean(0))
            if before:
                run["before_blowup"] = np.mean(before, 0).tolist()
                run["same_leads_finite"] = np.mean(same, 0).tolist() if same else None
            out["runs"][f"{kind}:{name}"] = run
    return out
