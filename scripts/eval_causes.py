"""Cause analysis of a checkpoint: error decomposition and eval-time ablations.

    python scripts/eval_causes.py --checkpoint runs/wb2_g1/checkpoint.pt --data-dir DATA \
        [--inits 31] [--lead-steps 12] [--variants no_denoiser composite ...] [--out causes.json]

See climate_wm/eval/causes.py: total error = autoencoder floor (rec) + latent
dynamics error (dyn), and the same rollout with parts switched off (paired
over initialisations). scripts/report_causes.py compares arms.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.eval import causes
from climate_wm.eval.graphcast_metrics import HEADLINE_KEYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None, help='replaces "{data_dir}" in the config')
    parser.add_argument("--inits", type=int, default=31, help="evenly spaced validation initialisations")
    parser.add_argument("--lead-steps", type=int, default=12, help="6 h steps (12 = 72 h)")
    parser.add_argument("--variants", nargs="*", default=None,
                        help=f"ablations (default: all that apply): no_denoiser, {', '.join(causes.COMPOSITE_PARTS)}")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    if args.data_dir:
        cfg = pipeline.substitute_data_dir(cfg, args.data_dir)
    pipeline.apply_overrides(cfg, args.set)
    cfg["eval"]["lead_steps"] = args.lead_steps
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train_ds, val_ds, _ = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    model.load_state_dict(state)
    variants = causes.available_variants(model) if args.variants is None else args.variants
    indices = np.linspace(0, len(val_ds) - 1, min(args.inits, len(val_ds))).astype(int)
    first, last = val_ds.times[val_ds.indices[indices[0]]], val_ds.times[val_ds.indices[indices[-1]]]
    print(f"{len(indices)} initialisations ({first} .. {last}), variants {variants}")
    res = causes.error_decomposition(model, val_ds, indices, stats, args.lead_steps, variants,
                                     device=args.device, batch_size=args.batch_size, log=print)
    res.update(experiment=cfg.get("experiment", ""), checkpoint=args.checkpoint,
               period=[str(first), str(last)])
    out = args.out or os.path.join(cfg["output_dir"], "causes.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    j = {v: i for i, v in enumerate(res["variables"])}
    for key in [k for k in HEADLINE_KEYS if k in j]:
        for li in sorted({0, len(res["lead_h"]) // 3, len(res["lead_h"]) - 1}):
            r = lambda name: res["rmse"][name][li][j[key]]
            line = (f"{key:28s} {res['lead_h'][li]:4d}h total {r('full'):.4g} rec {r('rec'):.4g} "
                    f"dyn {r('dyn'):.4g} rec share {res['rec_share'][li][j[key]]:.2f}")
            for name, ch in res["variant_change"].items():
                line += f" | {name} {100 * ch['mean'][li][j[key]]:+.1f}%"
            print(line)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
