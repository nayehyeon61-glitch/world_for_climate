"""Config -> data / model / training / GraphCast-style evaluation.

Shared by scripts/train.py, scripts/evaluate.py and scripts/smoke_test.py.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import graphcast_mapping as gm
from .data.era5_lazy_loader import GraphCastWindowDataset, open_era5
from .data.normalization import NormalizationStats, compute_stats
from .data.synthetic import make_synthetic_era5
from .eval.climatology import Climatology
from .eval.ensemble_metrics import EnsembleMetrics
from .eval.graphcast_metrics import GraphCastMetrics
from .geometry.shells import ShellSpec
from .models.world_model import ClimateWorldModel, WorldModelConfig
from .train.trainer import StageConfig, train_stage


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_layout(cfg: dict) -> gm.ChannelLayout:
    task = gm.get_task(cfg["task"]["name"], cfg["task"].get("pressure_levels"))
    shells = ShellSpec.from_config(cfg.get("shells"), task.pressure_levels)
    return gm.ChannelLayout(task, shells)


def open_dataset(cfg: dict):
    data = cfg["data"]
    if data.get("source", "era5") == "synthetic":
        return make_synthetic_era5(**data.get("synthetic", {}))
    with open(data["era5_config"]) as f:
        return open_era5(yaml.safe_load(f))


def build_datasets(cfg: dict, layout: gm.ChannelLayout, ds,
                   stats: Optional[NormalizationStats] = None):
    data = cfg["data"]
    train_steps = max(s.get("rollout_steps", 1) for s in cfg["train"]["stages"])
    split = lambda name: data[name]
    if stats is None:
        raw = GraphCastWindowDataset(ds, layout, None, train_steps, **split("train"))
        n = len(raw)
        idx = np.linspace(0, n - 1, min(n, data.get("stats_samples", 64))).astype(int)
        stats = compute_stats(raw, idx)
    train = GraphCastWindowDataset(ds, layout, stats, train_steps, **split("train"))
    lead = cfg["eval"]["lead_steps"]
    val = GraphCastWindowDataset(ds, layout, stats, lead, **split("val"),
                                 init_hours=cfg["eval"].get("init_hours"))
    return train, val, stats


def build_model(cfg: dict, layout: gm.ChannelLayout, dataset) -> ClimateWorldModel:
    return ClimateWorldModel(WorldModelConfig.from_dict(cfg["model"]), layout,
                             dataset.lat, dataset.lon)


def train(cfg: dict, model: ClimateWorldModel, train_ds, device="cpu",
          log=print) -> Dict[str, list]:
    loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=True, num_workers=cfg["train"].get("num_workers", 0),
                        drop_last=True)
    histories = {}
    for stage in cfg["train"]["stages"]:
        sc = StageConfig(**stage)
        if sc.name == "stochastic" and model.denoiser is None:
            log("[stochastic] skipped: transition has no diffusion residual")
            continue
        histories[sc.name] = train_stage(model, loader, sc, device,
                                         cfg["train"].get("log_every", 10), log)
    return histories


def save_checkpoint(path: str, cfg: dict, model: ClimateWorldModel,
                    stats: NormalizationStats) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"config": cfg, "model": model.state_dict(),
                "stats": {"mean": stats.mean, "std": stats.std,
                          "diff_std": stats.diff_std}}, path)


def load_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt["config"], ckpt["model"], NormalizationStats(**ckpt["stats"])


def build_climatology(cfg: dict, train_ds) -> Climatology:
    """WeatherBench-2 climatology if configured, else from the training windows."""
    path = cfg["eval"].get("climatology_zarr")
    if path:
        return Climatology.from_weatherbench2(path, train_ds.layout.targets)
    n = len(train_ds)
    idx = range(0, n, max(1, cfg["eval"].get("climatology_stride", 1)))
    return Climatology.from_window_dataset(train_ds, idx,
                                           cfg["eval"].get("climatology_window_days", 3))


def persistence_forecast(batch: Dict[str, torch.Tensor], layout: gm.ChannelLayout,
                         clim_norm: torch.Tensor) -> torch.Tensor:
    """B0: last input frame (normalised) for every lead; climatology for
    target channels that are not inputs (precipitation in *_operational)."""
    b, lead = batch["targets"].shape[:2]
    pred = clim_norm.clone()
    last = batch["inputs"][:, -1]
    for j, t in enumerate(layout.target_to_dynamic):
        pred[:, :, t] = last[:, j, None]
    return pred


@torch.no_grad()
def evaluate(cfg: dict, model: ClimateWorldModel, val_ds, stats: NormalizationStats,
             climatology: Climatology, device="cpu", max_batches: Optional[int] = None
             ) -> Dict[str, object]:
    """GraphCast protocol on `val_ds`: lat-weighted RMSE (eq. 20/21) and ACC
    (eq. 29) in physical units for the model and the B0 baselines; CRPS /
    spread-skill for the ensemble."""
    layout = val_ds.layout
    lead = cfg["eval"]["lead_steps"]
    members = cfg["eval"].get("members", 1)
    hours = [6 * (k + 1) for k in range(lead)]
    metrics = {name: GraphCastMetrics(layout.targets, val_ds.lat, hours)
               for name in ("model", "persistence", "climatology")}
    ens = EnsembleMetrics(layout.targets, val_ds.lat, hours) if members > 1 else None
    loader = DataLoader(val_ds, batch_size=cfg["eval"].get("batch_size", 2))
    model.eval()
    step_s = 6 * 3600
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        valid = batch["init_time"].cpu().numpy()[:, None] + step_s * np.arange(1, lead + 1)
        clim = climatology(valid)  # physical [B, L, C, H, W]
        target = stats.unnormalize(batch["targets"].cpu(), layout.targets)
        ens_pred = model.rollout(batch, lead, members=members).cpu()
        ens_phys = stats.unnormalize(ens_pred, layout.targets)
        metrics["model"].update(ens_phys.mean(0), target, clim)
        if ens is not None:
            ens.update(ens_phys, target)
        clim_norm = torch.from_numpy(stats.normalize(clim.numpy(), layout.targets))
        pers = persistence_forecast({k: v.cpu() for k, v in batch.items()}, layout, clim_norm)
        metrics["persistence"].update(stats.unnormalize(pers, layout.targets), target, clim)
        metrics["climatology"].update(clim, target, clim)
    out = {"metrics": metrics}
    if ens is not None:
        out["ensemble"] = ens
    return out


def results_to_json(results: Dict[str, object], path: str) -> None:
    payload = {name: m.table() for name, m in results["metrics"].items()}
    if "ensemble" in results:
        e = results["ensemble"].compute()
        keys = [c.key for c in results["ensemble"].channels]
        payload["ensemble"] = {
            name: {k: values[:, j].tolist() for j, k in enumerate(keys)}
            for name, values in e.items()}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
