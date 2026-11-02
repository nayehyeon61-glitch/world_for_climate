"""Staged training (research design section 7).

    Stage 1 repr        Shell-GNN autoencoder               encoder, decoder
    Stage 2 dynamics    deterministic transition + rollout  encoder, decoder, lie, residual
    Stage 3 stochastic  DIAMOND residual diffusion          denoiser
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from ..models.world_model import ClimateWorldModel

DEFAULT_TRAINABLE = {
    "repr": ("encoder", "decoder"),
    "dynamics": ("encoder", "decoder", "lie", "residual"),
    "stochastic": ("denoiser",),
    "theory": ("theory",),
}
MODULES = ("encoder", "decoder", "lie", "residual", "denoiser", "theory")


@dataclasses.dataclass
class StageConfig:
    name: str
    steps: int
    lr: float = 3e-4
    weight_decay: float = 1e-2  # DIAMOND denoiser optimiser
    warmup_steps: int = 0
    rollout_steps: int = 1
    grad_clip: float = 1.0
    weights: Dict[str, float] = dataclasses.field(default_factory=dict)
    trainable: Optional[Sequence[str]] = None


def set_trainable(model: ClimateWorldModel, names: Sequence[str]) -> List[torch.nn.Parameter]:
    params = []
    for name in MODULES:
        module = getattr(model, name)
        if module is None:
            continue
        flag = name in names
        for p in module.parameters():
            p.requires_grad_(flag)
            if flag:
                params.append(p)
    return params


def _forever(loader: DataLoader) -> Iterator:
    while True:
        yield from loader


def _to(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


@dataclasses.dataclass
class EngineConfig:
    """Large-grid options (train.engine in the config)."""
    amp: Optional[str] = None          # "bf16": autocast matmuls (Lie algebra + losses stay fp32)
    accum_steps: int = 1               # micro-batches per optimizer step
    grad_checkpointing: bool = False   # recompute GNN activations in backward
    save_every: int = 0                # optimizer steps between resumable checkpoints (0: stage end)


def _autocast(device, amp: Optional[str]):
    if amp is None:
        return contextlib.nullcontext()
    if amp != "bf16":
        raise ValueError(f"amp must be null or 'bf16', got {amp!r}")
    return torch.autocast(torch.device(device).type, dtype=torch.bfloat16)


def _memory(device) -> str:
    if torch.device(device).type == "cuda":
        return f" mem={torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f}GB"
    return ""


def train_stage(model: ClimateWorldModel, loader: DataLoader, cfg: StageConfig,
                device="cpu", log_every: int = 10,
                log: Callable[[str], None] = print,
                engine: Optional[EngineConfig] = None,
                resume: Optional[dict] = None,
                on_save: Optional[Callable[[dict], None]] = None) -> List[Dict[str, float]]:
    """Trains one stage. `resume` ({step, optimizer, scheduler, history}) continues
    an interrupted stage; `on_save(progress)` receives that dict every
    `engine.save_every` optimizer steps."""
    engine = engine or EngineConfig()
    if cfg.name == "stochastic" and model.denoiser is None:
        raise ValueError("The stochastic stage needs a *_diffusion transition.")
    params = set_trainable(model, cfg.trainable or DEFAULT_TRAINABLE[cfg.name])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_lambda(step):
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = step / max(cfg.steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    history, first = [], 0
    if resume is not None:
        opt.load_state_dict(resume["optimizer"])
        sched.load_state_dict(resume["scheduler"])
        history, first = list(resume["history"]), int(resume["step"])
        log(f"[{cfg.name}] resuming at step {first}")
    batches = _forever(loader)
    model.train()
    if cfg.name == "stochastic" and not bool(model.residual_calibrated):
        with _autocast(device, engine.amp):
            model.calibrate_residual_scale(_to(next(batches), device))
        log(f"[stochastic] residual scale = {model.residual_scale.tolist()}")
    t0, samples = time.time(), 0
    for step in range(first, cfg.steps):
        opt.zero_grad(set_to_none=True)
        logs: Dict[str, float] = {}
        loss_total = 0.0
        for _ in range(engine.accum_steps):
            batch = _to(next(batches), device)
            with _autocast(device, engine.amp):
                loss, micro = model.losses(batch, cfg.name, cfg.weights, cfg.rollout_steps)
            (loss / engine.accum_steps).backward()
            loss_total += float(loss.detach()) / engine.accum_steps
            for k, v in micro.items():
                logs[k] = logs.get(k, 0.0) + v / engine.accum_steps
            samples += batch["targets"].shape[0]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        logs.update(loss=loss_total, grad_norm=float(grad_norm), step=step)
        history.append(logs)
        if log_every and (step % log_every == 0 or step == cfg.steps - 1):
            terms = " ".join(f"{k}={v:.4g}" for k, v in logs.items()
                             if k not in ("step", "grad_norm"))
            elapsed = time.time() - t0
            log(f"[{cfg.name}] step {step:>5} {terms} ({elapsed:.1f}s, "
                f"{samples / max(elapsed, 1e-9):.2f} samples/s{_memory(device)})")
        if on_save is not None and engine.save_every and (step + 1) % engine.save_every == 0 \
                and step + 1 < cfg.steps:
            on_save({"stage": cfg.name, "step": step + 1, "optimizer": opt.state_dict(),
                     "scheduler": sched.state_dict(), "history": list(history)})
    model.eval()
    return history
