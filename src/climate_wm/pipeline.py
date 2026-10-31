"""Config -> data / model / training / GraphCast-style evaluation.

Shared by scripts/train.py, scripts/evaluate.py and scripts/smoke_test.py.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import xarray as xr
import yaml
from torch.utils.data import DataLoader, Subset

from .data import graphcast_mapping as gm
from .data.era5_lazy_loader import GraphCastWindowDataset, open_era5, standardize
from .data.normalization import NormalizationStats, compute_stats
from .data.sources import HourlyArchive
from .data.synthetic import make_synthetic_era5
from .eval.climatology import Climatology
from .eval.ensemble_metrics import EnsembleMetrics
from .eval.graphcast_metrics import GraphCastMetrics
from .geometry.shells import ShellSpec
from .models.world_model import ClimateWorldModel, WorldModelConfig
from .train.trainer import EngineConfig, StageConfig, train_stage


def load_config(path: str, data_dir: Optional[str] = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return substitute_data_dir(cfg, data_dir) if data_dir else cfg


def substitute_data_dir(cfg, data_dir: str):
    """Replaces "{data_dir}" in every string of the config."""
    if isinstance(cfg, dict):
        return {k: substitute_data_dir(v, data_dir) for k, v in cfg.items()}
    if isinstance(cfg, list):
        return [substitute_data_dir(v, data_dir) for v in cfg]
    if isinstance(cfg, str):
        return cfg.replace("{data_dir}", data_dir)
    return cfg


def build_layout(cfg: dict) -> gm.ChannelLayout:
    task = gm.get_task(cfg["task"]["name"], cfg["task"].get("pressure_levels"))
    shells = ShellSpec.from_config(cfg.get("shells"), task.pressure_levels)
    return gm.ChannelLayout(task, shells)


def apply_overrides(cfg: dict, overrides) -> dict:
    """KEY=VALUE strings with dotted keys; values are parsed as YAML."""
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return cfg


def merge_config(base: dict, update: dict) -> dict:
    """Recursive dict update (returns a new dict)."""
    out = dict(base)
    for k, v in update.items():
        out[k] = merge_config(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def open_dataset(cfg: dict):
    data = cfg["data"]
    levels = list(cfg["task"].get("pressure_levels")
                  or gm.TASKS[cfg["task"]["name"]].pressure_levels)
    source = data.get("source", "era5")
    if source == "synthetic":
        return make_synthetic_era5(**data.get("synthetic", {}))
    if source == "zarr":  # e.g. a WeatherBench-2 subset (scripts/download_weatherbench2.py)
        ds = standardize(xr.open_zarr(data["path"]), hours=data.get("hours", (0, 6, 12, 18)))
        if data.get("time_range"):
            ds = ds.sel(time=slice(*data["time_range"]))
        if "level" in ds.dims:
            ds = ds.sel(level=levels)
        return ds.load() if data.get("load_into_memory") else ds
    if source == "era5_hourly":  # indexed reader for the hourly 0.25 deg archive
        archive_cfg = {}
        if data.get("era5_config"):
            with open(data["era5_config"]) as f:
                archive_cfg = yaml.safe_load(f)
        archive_cfg.update(data.get("archive", {}))  # per-experiment overrides, e.g. coarsen
        if data.get("time_range"):
            archive_cfg["time_range"] = data["time_range"]
        return HourlyArchive(archive_cfg)
    with open(data["era5_config"]) as f:
        return open_era5(yaml.safe_load(f))


def build_datasets(cfg: dict, layout: gm.ChannelLayout, ds,
                   stats: Optional[NormalizationStats] = None):
    data = cfg["data"]
    train_steps = max(s.get("rollout_steps", 1) for s in cfg["train"]["stages"])
    split = lambda name: data[name]
    stats_path = data.get("stats_path")
    if stats is None and stats_path and os.path.exists(stats_path):
        stats = NormalizationStats.load(stats_path)
    if stats is None:
        raw = GraphCastWindowDataset(ds, layout, None, train_steps, **split("train"))
        n = len(raw)
        idx = np.linspace(0, n - 1, min(n, data.get("stats_samples", 64))).astype(int)
        stats = compute_stats(raw, idx, num_workers=data.get("stats_workers", 0), log=print)
        if stats_path:
            os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
            stats.save(stats_path)
    train = GraphCastWindowDataset(ds, layout, stats, train_steps, **split("train"))
    lead = cfg["eval"]["lead_steps"]
    val = GraphCastWindowDataset(ds, layout, stats, lead, **split("val"),
                                 init_hours=cfg["eval"].get("init_hours"))
    return train, val, stats


def build_model(cfg: dict, layout: gm.ChannelLayout, dataset) -> ClimateWorldModel:
    if cfg["model"].get("type") == "physics":  # stage 9: primitive-equation core
        from .models.physics_world_model import PhysicsModelConfig, PhysicsWorldModel
        model = PhysicsWorldModel(PhysicsModelConfig.from_dict(cfg["model"]), layout,
                                  dataset.lat, dataset.lon)
    else:
        model = ClimateWorldModel(WorldModelConfig.from_dict(cfg["model"]), layout,
                                  dataset.lat, dataset.lon)
    stats = getattr(dataset, "stats", None)
    if stats is not None:
        model.set_normalization(stats)  # a loaded checkpoint carries the same buffers
    return model


def _worker_init(_):
    torch.set_num_threads(1)
    try:
        import dask
        dask.config.set(scheduler="synchronous")  # no thread pools inside workers
    except ImportError:
        pass


def make_loader(cfg: dict, dataset, device="cpu") -> DataLoader:
    tc = cfg["train"]
    workers = tc.get("num_workers", 0)
    extra = dict(persistent_workers=True, prefetch_factor=tc.get("prefetch_factor", 2),
                 worker_init_fn=_worker_init) if workers > 0 else {}
    return DataLoader(dataset, batch_size=tc["batch_size"], shuffle=True,
                      num_workers=workers, drop_last=True,
                      pin_memory=torch.device(device).type == "cuda", **extra)


def train(cfg: dict, model: ClimateWorldModel, train_ds, device="cpu",
          log=print, histories: Optional[Dict[str, list]] = None,
          on_stage_end: Optional[Callable] = None,
          progress: Optional[dict] = None,
          on_progress: Optional[Callable[[dict], None]] = None) -> Dict[str, list]:
    """Runs the configured stages in order. Stages already in `histories`
    (from a resumed checkpoint) are skipped; `on_stage_end(histories)` runs
    after each finished stage. `progress` resumes an interrupted stage and
    `on_progress(progress)` is called every `train.engine.save_every` steps."""
    engine = EngineConfig(**cfg["train"].get("engine", {}))
    model.set_grad_checkpointing(engine.grad_checkpointing)
    loader = make_loader(cfg, train_ds, device)
    histories = dict(histories or {})
    for stage in cfg["train"]["stages"]:
        sc = StageConfig(**stage)
        if sc.name in histories:
            log(f"[{sc.name}] skipped: already trained (resumed checkpoint)")
            continue
        if sc.name == "stochastic" and model.denoiser is None:
            log("[stochastic] skipped: transition has no diffusion residual")
            continue
        if sc.name == "theory" and model.theory is None:
            log("[theory] skipped: transition is not lie_program")
            continue
        resume = progress if progress and progress.get("stage") == sc.name else None
        histories[sc.name] = train_stage(model, loader, sc, device,
                                         cfg["train"].get("log_every", 10), log,
                                         engine=engine, resume=resume,
                                         on_save=on_progress)
        if on_stage_end is not None:
            on_stage_end(histories)
    return histories


def save_checkpoint(path: str, cfg: dict, model: ClimateWorldModel,
                    stats: NormalizationStats,
                    histories: Optional[Dict[str, list]] = None,
                    progress: Optional[dict] = None) -> None:
    """`progress` (optimizer / scheduler state of an unfinished stage) makes
    the checkpoint resumable mid-stage."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"config": cfg, "model": model.state_dict(),
                "stats": {"mean": stats.mean, "std": stats.std,
                          "diff_std": stats.diff_std},
                "histories": histories or {}, "progress": progress}, tmp)
    os.replace(tmp, path)  # never leave a half-written checkpoint behind


def load_checkpoint(path: str, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt["config"], ckpt["model"], NormalizationStats(**ckpt["stats"])


def load_matching_state(model: ClimateWorldModel, path: str, log=print) -> None:
    """Initialises `model` from a checkpoint trained on another grid / mesh
    (e.g. 1 deg -> 0.25 deg): copies parameters and non-geometry buffers whose
    shapes match; graph buffers are rebuilt for the new grid and the residual
    scale is recalibrated."""
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    own = model.state_dict()
    skip_prefix = ("graphs.",)
    skip_names = ("residual_scale", "residual_calibrated")
    copied, skipped = 0, []
    for k, v in state.items():
        if k.startswith(skip_prefix) or k in skip_names:
            continue
        if k in own and own[k].shape == v.shape:
            own[k] = v
            copied += 1
        else:
            skipped.append(k)
    # A lie_program model's programmer has the Lie transition's architecture
    # (two latent frames in, one algebra element out): warm-start it from the
    # trained transition when the checkpoint has none of its own.
    warm = 0
    if model.theory is not None and not any(k.startswith("theory.proposer.") for k in state):
        for k, v in state.items():
            if k.startswith("lie."):
                tk = "theory.proposer." + k[len("lie."):]
                if tk in own and own[tk].shape == v.shape:
                    own[tk] = v
                    warm += 1
    model.load_state_dict(own)
    log(f"[init] copied {copied} tensors from {path}; skipped {len(skipped)} "
        f"(shape or name mismatch){': ' + ', '.join(skipped[:5]) if skipped else ''}"
        + (f"; programmer warm-started from the Lie transition ({warm} tensors)" if warm else ""))


def build_climatology(cfg: dict, train_ds) -> Climatology:
    """WeatherBench-2 climatology if configured, else from the training windows."""
    path = cfg["eval"].get("climatology_zarr")
    if path:  # climatology_load: false reads frames lazily (0.25 deg is ~400 GB in memory)
        return Climatology.from_weatherbench2(path, train_ds.layout.targets,
                                              load=cfg["eval"].get("climatology_load", True),
                                              coarsen=cfg["eval"].get("climatology_coarsen", 1))
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
             climatology: Climatology, device="cpu", max_batches: Optional[int] = None,
             indices: Optional[Sequence[int]] = None,
             extra_forecasts: Optional[Dict[str, Callable]] = None,
             log=None, progress_path: Optional[str] = None) -> Dict[str, object]:
    """GraphCast protocol on `val_ds`: lat-weighted RMSE (eq. 20/21) and ACC
    (eq. 29) in physical units for the model and the B0 baselines; CRPS /
    spread-skill for the ensemble.

    `extra_forecasts` maps a name to fn(init_times [B] in s, lead) returning
    physical forecasts [B, L, C_tgt, H, W] (e.g. GraphCast from WeatherBench-2),
    scored on exactly the same initialisations and targets.

    `progress_path`: the accumulated scores (and the random state) are saved
    there after every batch, and an existing file is continued, so a long
    evaluation survives being killed.
    """
    layout = val_ds.layout
    lead = cfg["eval"]["lead_steps"]
    members = cfg["eval"].get("members", 1)
    hours = [6 * (k + 1) for k in range(lead)]
    # "model" is the ensemble mean; "model_member" scores a single sample so a
    # deterministic baseline is not compared only against a smoothed mean.
    names = ["model", *(["model_member"] if members > 1 else []),
             *(extra_forecasts or {}), "persistence", "climatology"]
    metrics = {name: GraphCastMetrics(layout.targets, val_ds.lat, hours) for name in names}
    ens = EnsembleMetrics(layout.targets, val_ds.lat, hours) if members > 1 else None
    order = list(range(len(val_ds))) if indices is None else list(indices)
    done = 0
    if progress_path is not None and os.path.exists(progress_path):
        saved = torch.load(progress_path, weights_only=False)
        metrics, ens, done = saved["metrics"], saved["ensemble"], saved["done"]
        torch.set_rng_state(saved["rng"])
        if log is not None:
            log(f"[eval] resumed after {done}/{len(order)} initialisations")
    data = Subset(val_ds, order[done:])
    loader = DataLoader(data, batch_size=cfg["eval"].get("batch_size", 2))
    model.eval()
    step_s = 6 * 3600
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        init = batch["init_time"].cpu().numpy()
        batch = {k: v.to(device) for k, v in batch.items()}
        valid = init[:, None] + step_s * np.arange(1, lead + 1)
        clim = climatology(valid)  # physical [B, L, C, H, W]
        target = stats.unnormalize(batch["targets"].cpu(), layout.targets)
        ens_pred = model.rollout(batch, lead, members=members).cpu()
        ens_phys = stats.unnormalize(ens_pred, layout.targets)
        metrics["model"].update(ens_phys.mean(0), target, clim)
        if ens is not None:
            metrics["model_member"].update(ens_phys[0], target, clim)
            ens.update(ens_phys, target)
        for name, fn in (extra_forecasts or {}).items():
            metrics[name].update(fn(init, lead), target, clim)
        clim_norm = torch.from_numpy(stats.normalize(clim.numpy(), layout.targets))
        pers = persistence_forecast({k: v.cpu() for k, v in batch.items()}, layout, clim_norm)
        metrics["persistence"].update(stats.unnormalize(pers, layout.targets), target, clim)
        metrics["climatology"].update(clim, target, clim)
        n = done + min((i + 1) * loader.batch_size, len(data))
        if progress_path is not None:
            torch.save({"metrics": metrics, "ensemble": ens, "done": n, "rng": torch.get_rng_state()},
                       progress_path + ".tmp")
            os.replace(progress_path + ".tmp", progress_path)
        if log is not None:
            log(f"[eval] {n}/{len(order)} initialisations")
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
