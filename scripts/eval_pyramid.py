"""Stage 4 evaluation of a pyramid checkpoint: energy spectrum + per-level transfer.

    python scripts/eval_pyramid.py --checkpoint runs/wb2_pyramid/checkpoint.pt \
        --data-dir DATA [--stride 4] [--min-gap-days 7] [--out results.json]

See climate_wm/eval/pyramid.py. The 72-120 h skill (prediction 3) comes from
scripts/compare_graphcast.py; scripts/report_pyramid.py joins the two.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch

from climate_wm import pipeline
from climate_wm.eval import pyramid as pyr_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=None, help='replaces "{data_dir}" in the config')
    parser.add_argument("--stride", type=int, default=4, help="every n-th validation initialisation")
    parser.add_argument("--min-gap-days", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()

    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    if args.data_dir:
        cfg = pipeline.substitute_data_dir(cfg, args.data_dir)
    pipeline.apply_overrides(cfg, args.set)
    cfg["eval"]["lead_steps"] = max(1, int(cfg["eval"].get("lead_steps", 1)))
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train_ds, val_ds, _ = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    model.load_state_dict(state)
    if model.graphs.pyramid is None:
        parser.error("the checkpoint has no pyramid latent (mesh.pyramid_levels)")
    indices = np.arange(len(val_ds))[::args.stride]
    print(f"{len(indices)} pairs ({val_ds.times[val_ds.indices[indices[0]]]} .. "
          f"{val_ds.times[val_ds.indices[indices[-1]]]})")

    pairs = pyr_eval.collect_pairs(model, val_ds, indices, args.device, args.batch_size, print)
    parts = pyr_eval.explain_all(model, pairs, args.device, args.batch_size, print)
    shells = [s.name for s in layout.shells.shells]
    results = {
        "experiment": cfg.get("experiment", ""),
        "checkpoint": args.checkpoint,
        "levels": list(model.graphs.pyramid.levels),
        "rot_levels": list(model.graphs.pyramid.rot_levels),
        "pairs": len(pairs),
        "energy": pyr_eval.level_energy(parts, pyr_eval.patch_latitudes(model), shells),
    }
    analogs = pyr_eval.find_analogs(pairs, 24.0 * args.min_gap_days)
    results["transfer"] = pyr_eval.level_transfer(model, pairs, parts, analogs,
                                                  args.device, args.batch_size)
    print(pyr_eval.markdown(results))
    out = args.out or os.path.join(cfg["output_dir"], "pyramid_eval.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
