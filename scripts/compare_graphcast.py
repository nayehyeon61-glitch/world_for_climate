"""Scores our model and GraphCast with the same GraphCast protocol.

    python scripts/compare_graphcast.py --checkpoint runs/wb2_64x32/checkpoint.pt \
        --data-dir data/wb2_64x32 [--init-stride 4] [--lead-steps 40]

GraphCast forecasts come from WeatherBench-2 (`graphcast.zarr` written by
scripts/download_weatherbench2.py, regridded by WB2 to the evaluation grid).
Both are evaluated on the same initialisations, lead times, ERA5 targets,
climatology and latitude weights: RMSE (eq. 20), RMSE_trad (eq. 21),
ACC (eq. 29) and RMSE / ACC skill scores of our model relative to GraphCast.
"""

import argparse
import json
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import numpy as np
import torch
import xarray as xr

from climate_wm import pipeline
from climate_wm.eval.graphcast_metrics import (HEADLINE_KEYS, acc_skill_score,
                                               format_scorecard, rmse_skill_score)


class GraphCastForecasts:
    """fn(init_times [B] seconds, lead) -> physical forecasts [B, L, C, H, W]."""

    def __init__(self, path, channels):
        ds = xr.open_zarr(path)
        ds = ds.rename({k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items()
                        if k in ds.dims}).sortby("lat").sortby("lon")
        self.ds, self.channels = ds, channels
        lead = ds.prediction_timedelta.values
        if np.issubdtype(lead.dtype, np.timedelta64):
            lead = lead / np.timedelta64(1, "h")
        self.lead_hours = np.asarray(lead, np.float64)
        self.inits = ds.time.values.astype("datetime64[s]").astype(np.int64)

    def available(self, init_seconds):
        return np.isin(init_seconds, self.inits)

    def __call__(self, init_seconds, lead):
        want = 6.0 * np.arange(1, lead + 1)
        idx = [int(np.where(np.isclose(self.lead_hours, h))[0][0]) for h in want]
        times = np.asarray(init_seconds).astype("datetime64[s]")
        sub = self.ds.sel(time=times).isel(prediction_timedelta=idx)
        frames = []
        for c in self.channels:
            da = sub[c.name] if c.level is None else sub[c.name].sel(level=c.level)
            frames.append(np.asarray(
                da.transpose("time", "prediction_timedelta", "lat", "lon").values, np.float32))
        return torch.from_numpy(np.stack(frames, axis=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--lead-steps", type=int, default=None)
    parser.add_argument("--init-stride", type=int, default=1)
    parser.add_argument("--members", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg, state, stats = pipeline.load_checkpoint(args.checkpoint, args.device)
    cfg = pipeline.substitute_data_dir(cfg, args.data_dir)
    if args.lead_steps:
        cfg["eval"]["lead_steps"] = args.lead_steps
    if args.members:
        cfg["eval"]["members"] = args.members
    layout = pipeline.build_layout(cfg)
    ds = pipeline.open_dataset(cfg)
    train_ds, val_ds, _ = pipeline.build_datasets(cfg, layout, ds, stats)
    model = pipeline.build_model(cfg, layout, train_ds).to(args.device)
    model.load_state_dict(state)
    clim = pipeline.build_climatology(cfg, train_ds)
    gc = GraphCastForecasts(os.path.join(args.data_dir, "graphcast.zarr"), layout.targets)

    inits = np.array([val_ds.times[i].astype("datetime64[s]").astype(np.int64)
                      for i in val_ds.indices])
    usable = np.where(gc.available(inits))[0][::args.init_stride]
    print(f"{len(usable)} common initialisations "
          f"({val_ds.times[val_ds.indices[usable[0]]]} .. {val_ds.times[val_ds.indices[usable[-1]]]})")
    results = pipeline.evaluate(cfg, model, val_ds, stats, clim, args.device,
                                indices=usable, extra_forecasts={"graphcast": gc},
                                log=print)
    metrics = results["metrics"]
    print(format_scorecard(metrics, "model", "graphcast"))
    out = args.out or os.path.join(cfg["output_dir"], "compare_graphcast.json")
    pipeline.results_to_json(results, out)
    ours, theirs = metrics["model"].compute(), metrics["graphcast"].compute()
    index = {c.key: i for i, c in enumerate(layout.targets)}
    summary = {}
    for key in HEADLINE_KEYS:
        if key in index:
            j = index[key]
            summary[key] = {
                "lead_h": metrics["model"].lead_hours,
                "rmse_skill_vs_graphcast": rmse_skill_score(ours["rmse"][:, j], theirs["rmse"][:, j]).tolist(),
                "acc_skill_vs_graphcast": acc_skill_score(ours["acc"][:, j], theirs["acc"][:, j]).tolist(),
            }
    with open(out.replace(".json", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
