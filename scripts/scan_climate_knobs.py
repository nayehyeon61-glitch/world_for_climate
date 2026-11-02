#!/usr/bin/env python
"""Stage 16 cause scan: constant knob settings on the same 8 training windows (2018-19, 00 UTC), 15-day free
runs of N1. Which knobs move the blow-up day and the daily distance? (README_KO section 16)

    python scripts/scan_climate_knobs.py run --checkpoint runs/physics_n1/checkpoint.pt --data wb2_2020/era5.zarr \
        --calib runs/stage16/calib.json --out runs/stage16/knobscan.json \
        --settings base= "Q_upper-1=Q_upper:-1" "diffusion-1 Q_upper-1=diffusion:-1,Q_upper:-1"
    python scripts/scan_climate_knobs.py table runs/stage16/knobscan.json runs/stage16/knobscan2.json \
        --out docs/results/stage16_rain_game/knob_scan.md

A setting is NAME=knob:a,knob:a with a in [-1, 1] (models/knobs.py: multiplier 1 + 0.5 a; diffusion time
x 2^a, so a = +1 is weaker diffusion). Resumable per setting.
"""
from __future__ import annotations

import argparse
import json
import os
import types

import _path  # noqa: F401
import numpy as np
import torch

from climate_wm.models import knobs as K
from climate_wm.train.climate_game import GameConfig, Scorer, play

DAYS = 15


def parse(spec: str):
    name, _, body = spec.partition("=")
    v = torch.zeros(len(K.KNOBS))
    for item in filter(None, body.split(",")):
        k, x = item.split(":")
        v[K.KNOBS.index(k)] = float(x)
    return name, v


def run(args) -> None:
    import play_climate_game as P
    torch.set_num_threads(1)
    ns = types.SimpleNamespace(checkpoint=args.checkpoint, data=args.data, device="cpu", set=[])
    model, win = P.build(ns, 4 * DAYS, "train")
    sc, cfg = Scorer(model), GameConfig(days=DAYS)
    batch = win.batch(win.spread(16)[::2])
    calib = json.load(open(args.calib))
    ref = {"day": calib["day"], "term": calib["term"]}
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for spec in args.settings:
        name, v = parse(spec)
        if name in res:
            continue
        with torch.no_grad():
            ep = play(model, batch, lambda o, d: v.expand(o.shape[0], -1), cfg, sc, ref)
        bd, err = ep["blown_day"].tolist(), ep["err"].numpy()
        res[name] = {"setting": dict(zip(K.KNOBS, v.tolist())), "blown_days": bd,
                     "alive_mean": float(np.mean([d if d >= 0 else DAYS for d in bd])),
                     "err_day": [None if not np.isfinite(x) else float(x)
                                 for x in np.nanmean(np.where(np.isfinite(err), err, np.nan), 1)],
                     "return": float(ep["rew"].sum(0).mean())}
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"{name:36s} blown {sum(d >= 0 for d in bd)}/{len(bd)} alive {res[name]['alive_mean']:.1f} "
              f"return {res[name]['return']:.1f}", flush=True)


def table(args) -> None:
    res = {}
    for p in args.files:
        res.update(json.load(open(p)))
    base = res.get("base")
    L = ["# Stage 16 knob scan: constant settings, 15-day free runs of N1 on 8 training windows", "",
         "a = +-1 is the end of a knob's range (multiplier 1 + 0.5 a; diffusion time x 2^a, a = +1 = weaker "
         "diffusion). Alive days count 15 for runs that never blew up. Daily distance = the game's 5-day smoothed "
         "zonal-mean distance to ERA5 on day 8 (normalised units), mean over the runs still alive. The reading is "
         "in README_KO section 16." if base else "", "",
         "| setting | blown | alive days mean | day 8 distance | return |", "|---|---|---|---|---|"]
    order = sorted(res, key=lambda n: (n != "base", -res[n]["alive_mean"], -res[n]["return"]))
    for n in order:
        r = res[n]
        d8 = r["err_day"][7] if len(r["err_day"]) > 7 else None
        L.append(f"| {n} | {sum(d >= 0 for d in r['blown_days'])}/{len(r['blown_days'])} | {r['alive_mean']:.1f} | "
                 f"{'-' if d8 is None else f'{d8:.3f}'} | {r['return']:.1f} |")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--data", required=True)
    r.add_argument("--calib", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--settings", nargs="+", required=True)
    t = sub.add_parser("table")
    t.add_argument("files", nargs="+")
    t.add_argument("--out", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else table(args)


if __name__ == "__main__":
    main()
