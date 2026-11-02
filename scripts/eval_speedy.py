#!/usr/bin/env python
"""Stage 14 S0: a world-model checkpoint scored on the SPEEDY nature run.

The checkpoint keeps its own normalisation statistics (ERA5), so this is the
zero-shot domain gap; the climatology baseline comes from the SPEEDY training
years. Initialisations are spread evenly over the test year like the
GraphCast frame (31 by default).

    python scripts/eval_speedy.py --checkpoint runs/physics_n1/checkpoint.pt \
        --zarr $S/speedy_wb2/nature.zarr --out runs/stage14/n1_speedy1984.json
"""
from __future__ import annotations

import argparse
import os

import _path  # noqa: F401
import numpy as np
import torch

import pandas as pd

from climate_wm import pipeline
from climate_wm.eval.climatology import Climatology
from climate_wm.eval.graphcast_metrics import format_scorecard


class _MonthlyValues:
    """values[d, h] of a Climatology from monthly x hour-of-day means
    (the day-of-year table at 64x32 x 83 channels would hold 2-6 GB while built)."""

    def __init__(self, monthly: np.ndarray):
        self.monthly = monthly
        self.month = np.append(pd.date_range("2001-01-01", periods=365).month.values - 1, 11)

    def __getitem__(self, idx):
        d, h = idx
        return self.monthly[self.month[d], h]


def monthly_climatology(train_ds, stride: int) -> Climatology:
    hours = (0, 6, 12, 18)
    sums, counts = None, np.zeros((12, 4))
    for k in range(0, len(train_ds), stride):
        frames, _, times = train_ds.raw_window(k)
        for frame, t in zip(frames, times):
            ts = pd.Timestamp(t)
            if ts.hour not in hours:
                continue
            if sums is None:
                sums = np.zeros((12, 4) + frame.shape, np.float64)
            sums[ts.month - 1, hours.index(ts.hour)] += frame
            counts[ts.month - 1, hours.index(ts.hour)] += 1
    monthly = (sums / np.maximum(counts, 1)[..., None, None, None]).astype(np.float32)
    return Climatology(_MonthlyValues(monthly), hours, channels=train_ds.layout.targets)


def speedy_config(cfg: dict, zarr: str, train, val, lead_steps: int) -> dict:
    cfg = dict(cfg)
    data = dict(cfg["data"])
    data.update(source="zarr", path=zarr, load_into_memory=False,
                time_range=[train[0][:10], (np.datetime64(val[1][:10]) + np.timedelta64(15, "D")).astype(str)],
                train={"start": train[0], "end": train[1]}, val={"start": val[0], "end": val[1]})
    data.pop("stats_path", None)
    cfg["data"] = data
    ev = dict(cfg.get("eval", {}))
    ev.update(climatology_zarr=None, climatology_stride=8, lead_steps=lead_steps, members=1,
              init_hours=[0, 12])
    cfg["eval"] = ev
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--train", nargs=2, default=["1982-01-01T06:00", "1983-12-31T18:00"])
    ap.add_argument("--val", nargs=2, default=["1984-01-01", "1984-12-31T12:00"])
    ap.add_argument("--lead-steps", type=int, default=20)
    ap.add_argument("--n-inits", type=int, default=31)
    ap.add_argument("--clim-stride", type=int, default=8, help="training windows between climatology samples")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    cfg = speedy_config(cfg, args.zarr, args.train, args.val, args.lead_steps)
    for kv in args.set:
        key, value = kv.split("=", 1)
        node = cfg
        *path, last = key.split(".")
        for p in path:
            node = node.setdefault(p, {})
        import yaml
        node[last] = yaml.safe_load(value)
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train_ds, val_ds, _ = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    model.load_state_dict(state, strict=False)
    model.eval()
    clim = monthly_climatology(train_ds, args.clim_stride)
    idx = np.linspace(0, len(val_ds) - 1, min(args.n_inits, len(val_ds))).astype(int).tolist()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = pipeline.evaluate(cfg, model, val_ds, stats, clim, args.device, indices=idx,
                                progress_path=args.out + ".progress", log=print)
    print(format_scorecard(results["metrics"], "model", "persistence"))
    pipeline.results_to_json(results, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
