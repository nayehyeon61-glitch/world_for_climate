"""Staged training (research design section 7).

    Stage 1 repr        Shell-GNN autoencoder               encoder, decoder
    Stage 2 dynamics    deterministic transition + rollout  encoder, decoder, lie, residual
    Stage 3 stochastic  DIAMOND residual diffusion          denoiser
"""

from __future__ import annotations

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
}
MODULES = ("encoder", "decoder", "lie", "residual", "denoiser")


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


def train_stage(model: ClimateWorldModel, loader: DataLoader, cfg: StageConfig,
                device="cpu", log_every: int = 10,
                log: Callable[[str], None] = print) -> List[Dict[str, float]]:
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
    batches = _forever(loader)
    model.train()
    if cfg.name == "stochastic" and not bool(model.residual_calibrated):
        model.calibrate_residual_scale(_to(next(batches), device))
        log(f"[stochastic] residual scale = {model.residual_scale.tolist()}")
    history, t0 = [], time.time()
    for step in range(cfg.steps):
        batch = _to(next(batches), device)
        loss, logs = model.losses(batch, cfg.name, cfg.weights, cfg.rollout_steps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        logs.update(loss=float(loss), grad_norm=float(grad_norm), step=step)
        history.append(logs)
        if log_every and (step % log_every == 0 or step == cfg.steps - 1):
            terms = " ".join(f"{k}={v:.4g}" for k, v in logs.items()
                             if k not in ("step", "grad_norm"))
            log(f"[{cfg.name}] step {step:>5} {terms} ({time.time() - t0:.1f}s)")
    model.eval()
    return history
