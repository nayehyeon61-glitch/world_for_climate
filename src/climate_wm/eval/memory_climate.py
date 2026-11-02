"""Stage 10 phase 2: does the adaptive memory carry what climate prediction needs?

By the Mori-Zwanzig formalism, the variables a model does not resolve (sea
surface temperature, soil moisture, sea ice) act on the resolved atmosphere
as a memory with long timescales; predictability beyond about two weeks comes
from them. The test, in the subseasonal (S2S) protocol:

  spin-up: `spinup` 6 h steps (10 days) with the masked truth re-observed
  after every step, so the memory absorbs the innovations;
  free run: `weeks` x 28 steps without observations; the weekly means of the
  forecast are scored against the weekly means of ERA5 (GraphCast-weighted
  MSE on the hidden and all points, and RMSE / ACC per variable against the
  climatology of the same weeks).

Variants (train/dreamer_game.VARIANTS): the arm in full, its knock-outs (no
actor, no memory) and N1 with nudging (the reference: the same spin-up
without memory). A run that blows up is scored with the climatology from the
step it blew up, and counted. Every initialisation gets the same mask and the
same weeks in every variant, so the variants pair.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..train.dreamer_game import REFERENCE, VARIANTS, play_memory
from ..train.physics_game import attach_loss
from ..train.prediction_game import errors, sample_mask, target_mask
from .graphcast_metrics import HEADLINE_KEYS, acc_per_init, normalized_latitude_weights, weighted_mean

WEEK = 28  # 6 h steps


def _bootstrap(rel: np.ndarray, seed: int) -> Dict[str, list]:
    idx = np.random.default_rng(seed).integers(0, len(rel), (2000, len(rel)))
    means = rel[idx].mean(1)
    return {"mean": rel.mean(0).tolist(), "lo": np.percentile(means, 2.5, 0).tolist(),
            "hi": np.percentile(means, 97.5, 0).tolist()}


@torch.no_grad()
def evaluate_memory_climate(model, ctrl, base, dataset, indices: Sequence[int], kinds: Sequence[str],
                            climatology, stats, spinup: int = 40, weeks: int = 4, week: int = WEEK, device="cpu",
                            variants: Sequence[str] = ("full", "no_actor", "no_memory"), seed: int = 0,
                            region_fraction: float = 0.25, cap: float = 10.0, log=None,
                            progress_path: Optional[str] = None) -> Dict[str, object]:
    """Weekly-mean scores of the free run after a spin-up, per kind (module
    docstring). `dataset` must hold spinup + weeks * week target steps (6 h). With
    `progress_path`, every finished initialisation is saved there and an
    existing file is continued."""
    layout = model.layout
    targets = list(layout.targets)
    horizon = spinup + weeks * week
    variants = [REFERENCE] + [v for v in variants if v != REFERENCE]
    lat_w = torch.tensor(normalized_latitude_weights(dataset.lat), dtype=torch.float32)
    keys = [k for k in HEADLINE_KEYS if k in [c.key for c in targets]]
    kidx = [[c.key for c in targets].index(k) for k in keys]
    attach_loss(model)  # the GraphCast weights of the hidden / all-point scores
    model.eval()
    if base is not None:
        base.eval()
    done: Dict[str, List[dict]] = {}
    if progress_path is not None and os.path.exists(progress_path):
        done = torch.load(progress_path, weights_only=False)
        if log is not None:
            log(f"[climate eval] resumed: {', '.join(f'{k} {len(v)}' for k, v in done.items())}")
    for kind in kinds:
        recs = done.setdefault(kind, [])
        for n, i in enumerate(indices):
            if n < len(recs):
                continue
            item = dataset[int(i)]
            batch = {k: v[None].to(device) for k, v in item.items()}
            gen = torch.Generator().manual_seed(seed + 1000 * n)
            if kind == "none":
                mask, label = torch.zeros(len(layout.dynamic), len(dataset.lat), len(dataset.lon)), "none"
            else:
                mask, label = sample_mask(layout, kind, dataset.lat, dataset.lon, gen, region_fraction)
            tmask = target_mask(model, mask[None].to(device))
            init = int(item["init_time"])
            valid = init + 6 * 3600 * np.arange(spinup + 1, horizon + 1)
            clim = climatology(valid[None])[0].float()  # physical [weeks * 28, C, H, W]
            clim_norm = torch.from_numpy(stats.normalize(clim.numpy(), targets)).float()
            weekly = lambda x: x.reshape(weeks, week, *x.shape[1:]).mean(1)
            truth = batch["targets"][0, spinup:horizon].float().cpu()
            truth_w, clim_w = weekly(truth), weekly(clim_norm)
            rec = {"label": label, "init": init}

            def score(name, pred_w):  # normalised weekly means [weeks, C, H, W]
                h, a = errors(model, pred_w.to(device), truth_w.to(device), tmask.expand(weeks, -1, -1, -1))
                phys = stats.unnormalize(pred_w, targets)
                obs = stats.unnormalize(truth_w, targets)
                clim_p = stats.unnormalize(clim_w, targets)
                mse = weighted_mean((phys - obs).square(), lat_w)[:, kidx]       # [weeks, keys]
                acc = acc_per_init(phys[None], obs[None], clim_p[None], lat_w)[0][:, kidx]
                rec[name] = {"hidden": h.cpu().tolist(), "all": a.cpu().tolist(),
                             "mse": mse.tolist(), "acc": acc.tolist()}

            score("climatology", clim_w)
            for name in variants:
                knock, act, which, rule = VARIANTS[name]
                m = model if which == "own" else base
                if m is None:
                    continue
                if m.memory is not None:
                    m.memory.knock = set(knock)
                sums = torch.zeros(weeks, *truth.shape[1:])
                blown = [False]

                def on_step(k, pred, dead):
                    if k < spinup:
                        return
                    blown[0] = blown[0] or bool(dead[0])
                    j = k - spinup
                    sums[j // week] += clim_norm[j] if blown[0] else pred[0].float().cpu()

                play_memory(m, ctrl if which == "own" else None, batch, mask[None], horizon, control=act,
                            obs_prob=1.0, rule=rule, gen=torch.Generator().manual_seed(seed + 7919 * n),
                            cap=cap, obs_steps=spinup, on_step=on_step)
                if m.memory is not None:
                    m.memory.knock = set()
                score(name, sums / week)
                rec[name]["blown"] = blown[0]
            recs.append(rec)
            if progress_path is not None:
                torch.save(done, progress_path + ".tmp")
                os.replace(progress_path + ".tmp", progress_path)
            if log is not None:
                log(f"[climate eval] {kind}: {n + 1}/{len(indices)} ({label})")
    out: Dict[str, object] = {"weeks": list(range(1, weeks + 1)), "spinup_days": spinup / 4,
                              "reference": REFERENCE, "variants": variants, "keys": keys, "kinds": {}}
    names = ["climatology", *variants]
    for kind in kinds:
        recs = done[kind]
        res = {"labels": [r["label"] for r in recs], "inits": [r["init"] for r in recs],
               "blowups": {v: int(sum(r[v].get("blown", False) for r in recs if v in r)) for v in variants},
               "mse": {}, "rmse": {}, "acc": {}, "change": {}}
        per = {}
        for v in names:
            if v not in recs[0]:
                continue
            for part in ("hidden", "all"):
                per[f"{v}_{part}"] = np.array([r[v][part] for r in recs])  # [N, weeks]
                res["mse"][f"{v}_{part}"] = per[f"{v}_{part}"].mean(0).tolist()
            mse = np.array([r[v]["mse"] for r in recs])  # [N, weeks, keys]
            res["rmse"][v] = {k: np.sqrt(mse[:, :, j]).mean(0).tolist() for j, k in enumerate(keys)}
            acc = np.array([r[v]["acc"] for r in recs])
            res["acc"][v] = {k: acc[:, :, j].mean(0).tolist() for j, k in enumerate(keys)}
        for v in names:
            for part in ("hidden", "all"):
                if v == REFERENCE or f"{v}_{part}" not in per:
                    continue
                ref = per[f"{REFERENCE}_{part}"]
                res["change"][f"{v}_{part}"] = _bootstrap((per[f"{v}_{part}"] - ref) / ref.mean(0), seed)
        out["kinds"][kind] = res
    return out
