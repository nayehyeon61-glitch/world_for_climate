"""Stage 13: blow-ups and the column's energy input under the E1 bounds (eval/energy.py).

    python scripts/diagnose_energy.py --checkpoint runs/stage10_Dnoactor/checkpoint.pt \
        --config configs/dreamer_wb2.yaml --data-dir DATA2020 --inits 12 --steps 20 --kinds none region \
        --out runs/stage10_Dnoactor/energy_2020.json [--settings none flux500 flux500_nowork]
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch
from train_active_obs import load_world_model

from climate_wm import pipeline
from climate_wm.eval.energy import SETTINGS, diagnose_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--inits", type=int, default=12)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--kinds", nargs="*", default=["none", "region"])
    parser.add_argument("--settings", nargs="*", default=list(SETTINGS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    update = pipeline.load_config(args.config, args.data_dir)
    cfg, model, data, stats = load_world_model(args.checkpoint, update, args.split, args.steps, args.device, args.set)
    indices = np.linspace(0, len(data) - 1, min(args.inits, len(data))).astype(int)
    settings = {k: SETTINGS[k] for k in args.settings}
    print(f"{len(indices)} initialisations, {args.steps} steps, kinds {args.kinds}, settings {list(settings)}")
    res = diagnose_energy(model, data, indices, args.steps, settings, kinds=args.kinds, device=args.device, log=print,
                          progress_path=args.out + ".progress")
    res.update(checkpoint=args.checkpoint, split=args.split)
    for name, r in res["runs"].items():
        ratio = r.get("err_ratio_finite")
        print(f"{name:24s} blow-ups {r['blowups']}/{len(res['inits'])}"
              + (f" | error ratio vs none (finite in both, n={r['finite_both']}) 24 h {ratio[3]:.3f} 72 h {ratio[min(11, len(ratio) - 1)]:.3f}"
                 if ratio else ""))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
