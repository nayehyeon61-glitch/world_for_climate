"""Stage 10 phase 2: weekly-mean (subseasonal) skill after a 10-day observed spin-up (eval/memory_climate.py).

    python scripts/eval_memory_climate.py --checkpoint runs/stage10_D/checkpoint.pt \
        --base runs/physics_n1/checkpoint.pt --config configs/dreamer_wb2.yaml --data-dir DATA \
        --out runs/stage10_D/climate_2020.json [--inits 12 --kinds none region --spinup-days 10 --weeks 4]

Initialisations are spread over 2020 (the windows need spin-up + free run
inside the data); every variant of every arm gets the same initialisations,
masks and weeks, so arms pair. Resumable (OUT.progress).
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch
from eval_dreamer import load_arm, usable

from climate_wm import pipeline
from climate_wm.eval.memory_climate import WEEK, evaluate_memory_climate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="stage 10 arm (OUT/checkpoint.pt, agent.pt beside it)")
    parser.add_argument("--base", required=True, help="the stage 9 model (the nudging reference)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--inits", type=int, default=12)
    parser.add_argument("--kinds", nargs="*", default=["none", "region"])
    parser.add_argument("--variants", nargs="*", default=["full", "no_actor", "no_memory"])
    parser.add_argument("--spinup-days", type=float, default=10.0)
    parser.add_argument("--weeks", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    update = pipeline.load_config(args.config, args.data_dir)
    spinup = int(round(4 * args.spinup_days))
    cfg, model, base, ctrl, data, stats = load_arm(args.checkpoint, args.base, update, "val",
                                                   spinup + args.weeks * WEEK, args.device)
    clim = pipeline.build_climatology(cfg, data)
    variants = usable(args.variants, model, ctrl)
    indices = np.linspace(0, len(data) - 1, min(args.inits, len(data))).astype(int)
    first, last = data.times[data.indices[indices[0]]], data.times[data.indices[indices[-1]]]
    print(f"{len(indices)} initialisations ({first} .. {last}), spin-up {spinup} steps, {args.weeks} weeks free, "
          f"kinds {args.kinds}, variants {variants}, actor {'yes' if ctrl is not None else 'no'}")
    res = evaluate_memory_climate(model, ctrl, base, data, indices, args.kinds, clim, stats, spinup=spinup,
                                  weeks=args.weeks, device=args.device, variants=variants, log=print,
                                  progress_path=args.out + ".progress")
    res.update(checkpoint=args.checkpoint, base=args.base, period=[str(first), str(last)], inits=len(indices))
    for kind, r in res["kinds"].items():
        line = f"{kind:7s} weekly hidden error vs nudging, weeks 1..{args.weeks}:"
        for name, ch in r["change"].items():
            if name.endswith("_hidden"):
                line += f" | {name[:-7]} " + " ".join(f"{100 * x:+.1f}%" for x in ch["mean"])
        print(line, "| blow-ups", r["blowups"])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
