"""GraphCast-protocol evaluation of a trained checkpoint.

    python scripts/evaluate.py --checkpoint runs/smoke/checkpoint.pt
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import torch

from climate_wm import pipeline
from climate_wm.eval.graphcast_metrics import format_scorecard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train_ds, val_ds, _ = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    model.load_state_dict(state)
    clim = pipeline.build_climatology(cfg, train_ds)
    results = pipeline.evaluate(cfg, model, val_ds, stats, clim, args.device,
                                args.max_batches)
    print(format_scorecard(results["metrics"], "model", "persistence"))
    path = os.path.join(cfg["output_dir"], "eval.json")
    pipeline.results_to_json(results, path)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
