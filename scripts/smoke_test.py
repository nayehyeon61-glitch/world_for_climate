"""P0 smoke test: synthetic data -> 3 training stages -> GraphCast evaluation.

    python scripts/smoke_test.py [--config configs/smoke.yaml] [--transition lie_residual]

Exercises the whole pipeline on CPU in a few minutes. The numbers only show
that training reduces the losses and that the evaluation runs; synthetic data
says nothing about real forecast skill.
"""

import argparse
import math
import os
import time

import _path  # noqa: F401  (adds src/ to sys.path)
import torch

from climate_wm import pipeline
from climate_wm.eval.graphcast_metrics import format_scorecard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(
        os.path.dirname(__file__), "..", "configs", "smoke.yaml"))
    parser.add_argument("--transition", default=None)
    parser.add_argument("--shell-coupling", default=None)
    parser.add_argument("--steps", type=int, default=None, help="override steps per stage")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cfg = pipeline.load_config(args.config)
    if args.transition:
        cfg["model"]["transition"] = args.transition
    if args.shell_coupling:
        cfg["model"]["lie"]["shell_coupling"] = args.shell_coupling
    if args.steps:
        for stage in cfg["train"]["stages"]:
            stage["steps"] = args.steps

    t0 = time.time()
    layout = pipeline.build_layout(cfg)
    print(layout.summary())
    ds = pipeline.open_dataset(cfg)
    train_ds, val_ds, stats = pipeline.build_datasets(cfg, layout, ds)
    print(f"train windows {len(train_ds)}, val windows {len(val_ds)}, grid {train_ds.grid_shape}")
    model = pipeline.build_model(cfg, layout, train_ds)
    print(model.graphs.summary())
    print(f"transition={cfg['model']['transition']} "
          f"coupling={cfg['model']['lie'].get('shell_coupling')} "
          f"parameters={sum(p.numel() for p in model.parameters()):,}")

    histories = pipeline.train(cfg, model, train_ds)
    for name, hist in histories.items():
        first, last = hist[0]["loss"], sum(h["loss"] for h in hist[-5:]) / min(5, len(hist))
        print(f"{name}: loss {first:.4f} -> {last:.4f}")
        assert all(math.isfinite(h["loss"]) for h in hist), f"non-finite loss in {name}"

    clim = pipeline.build_climatology(cfg, train_ds)
    results = pipeline.evaluate(cfg, model, val_ds, stats, clim, max_batches=4)
    print(format_scorecard(results["metrics"], "model", "persistence"))
    if "ensemble" in results:
        e = results["ensemble"].compute()
        print("ensemble spread/skill (z500-like mean over channels, per lead):",
              [round(float(x), 3) for x in e["spread_skill"].mean(1)])
    if model.lie is not None:
        batch = next(iter(torch.utils.data.DataLoader(val_ds, batch_size=1)))
        _, omegas = model.rollout(batch, 1, return_omega=True)
        omega = omegas[0][0]  # [S, P, 3]
        speed = omega.norm(dim=-1).mean(-1) * 6.371e6 / 21600
        print("mean |omega| per shell as equatorial speed [m/s]:",
              [round(float(s), 2) for s in speed])
    os.makedirs(cfg["output_dir"], exist_ok=True)
    pipeline.results_to_json(results, os.path.join(cfg["output_dir"], "smoke_eval.json"))
    print(f"smoke test OK in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
