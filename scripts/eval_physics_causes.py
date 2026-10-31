"""Stage 9 cause analysis of the physics world model (eval/physics_causes.py).

    python scripts/eval_physics_causes.py --checkpoint runs/physics_n1/checkpoint.pt --data-dir DATA \
        --year 2018 --out runs/physics_n1/causes_2018.json [--inits 8 --names no_pgf sun_off sun_off+no_heating]

Initialisations: 00 UTC on 1 January of `--year` and every `--spacing-days`
after (the 8 of the stage 9 N0 knock-outs for 2018), all in one batch; every
intervention and every model run on the same ones, so they pair. DATA must hold
the year plus three days. Resumable (OUT.progress).
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.data.era5_lazy_loader import GraphCastWindowDataset
from climate_wm.eval.physics_causes import evaluate_physics_causes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="stage 9 physics model (N0 or N1)")
    parser.add_argument("--data-dir", required=True, help="holds era5.zarr")
    parser.add_argument("--year", type=int, default=2018)
    parser.add_argument("--inits", type=int, default=8)
    parser.add_argument("--spacing-days", type=int, default=48)
    parser.add_argument("--steps", type=int, default=12, help="6 h leads (12 = 72 h)")
    parser.add_argument("--names", nargs="*", default=None, help="interventions and A+B pairs (default: all)")
    parser.add_argument("--cap", type=float, default=10.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, "cpu")
    y = args.year
    cfg = pipeline.merge_config(cfg, {"data": {"path": os.path.join(args.data_dir, "era5.zarr"),
                                               "time_range": [f"{y - 1}-12-31", f"{y + 1}-01-15"]}})
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    data = GraphCastWindowDataset(ds, layout, stats, args.steps, start=f"{y}-01-01", end=f"{y}-12-31T12:00")
    times = data.times[data.indices]
    starts = [np.datetime64(f"{y}-01-01T00") + np.timedelta64(args.spacing_days * j, "D") for j in range(args.inits)]
    indices = [int(np.where(times == t)[0][0]) for t in starts]
    print(f"{args.checkpoint}: {len(indices)} initialisations {starts[0]} .. {starts[-1]}, {args.steps} steps")
    res = evaluate_physics_causes(cfg["model"], state, stats, data, indices, names=args.names, steps=args.steps,
                                  cap=args.cap, device=args.device, log=lambda m: print(m, flush=True),
                                  progress_path=args.out + ".progress")
    res.update(checkpoint=args.checkpoint, year=y, init_times=[str(t) for t in starts])
    leads = res["leads_h"]
    for name, s in res["interventions"].items():
        if name == "full":
            continue
        ch = s["change_all"]["mean"]
        line = f"{name:34s} all-MSE change " + " ".join(f"{leads[l]}h {100 * ch[l]:+.0f}%" for l in (0, 3, 11)
                                                       if l < len(ch))
        if "pair" in s and "interaction_all" in s["pair"]:
            it = s["pair"]["interaction_all"]["mean"]
            line += " | interaction " + " ".join(f"{100 * it[l]:+.0f}%" for l in (0, 3, 11) if l < len(it))
        print(line, "| blow-ups", s["blowups"][-1])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
