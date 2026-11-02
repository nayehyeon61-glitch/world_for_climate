"""Policy improvement of the stochastic transition from verification reward.

The trained world model is reinforced on years it has not seen. The DIAMOND
denoiser is the policy: in latent state s_k = (z_{k-1}, z_k) with forcing
f_{k+1} it samples the residual a_k = r_k (the action), and
z_{k+1} = Exp_{zbar_{k+1}}(s * r_k). Each K-step rollout is verified against
the recorded ERA5 future:

    reward  R_k = - GraphCast-weighted MSE(D(z_{k+1}), X_{k+1})   (per member)
    return  G   = sum_k gamma^k R_k

M members per initial state form a group (as in GRPO): advantages are
A_m = (G_m - mean G) / std G inside the group, and the denoiser is regressed
on its own actions with weights M * softmax(A / beta) (advantage-weighted
regression; no gradient flows through the sampler). The supervised stochastic
stage loss on the same windows is kept as an anchor, so `rl_weight = 0` is
exactly the supervised fine-tuning control with the same data and steps.

Reanalysis does not react to the actions, so this is policy improvement from
verification feedback on logged trajectories, not interactive RL.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from ..models.latent import Latent
from ..models.stochastic_transition import LatentDiffusionSampler
from ..models.world_model import ClimateWorldModel
from .trainer import _forever, _to, set_trainable


@dataclasses.dataclass
class RefineConfig:
    mode: str = "rl"              # "rl" or "supervised" (the control: rl_weight = 0)
    steps: int = 150
    batch_size: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 10
    grad_clip: float = 1.0
    members: int = 4              # group size M
    horizon: int = 4              # rollout steps K (also the anchor's rollout length)
    gamma: float = 1.0
    beta: float = 1.0             # advantage temperature (large: imitate all members)
    rl_weight: float = 1.0
    anchor_weight: float = 1.0
    num_denoising_steps: Optional[int] = None
    awr_chunk: Optional[int] = None  # samples per AWR backward (None: batch_size); bounds memory
    log_every: int = 10
    seed: int = 0


def per_sample_error(model: ClimateWorldModel, pred, target):
    """GraphCast-weighted MSE per sample [N] (the training loss, unreduced)."""
    lf = model.loss_fn
    return (lf.per_channel(pred, target) * lf.channel_w).sum(-1)


@torch.no_grad()
def collect_rollouts(model: ClimateWorldModel, batch: Dict[str, torch.Tensor],
                     members: int, horizon: int, gamma: float = 1.0,
                     num_denoising_steps: Optional[int] = None):
    """Samples `members` trajectories per initial state.

    Returns the per-step conditioning and actions (members folded into the
    batch as b0m0, b0m1, ...), the returns G [B, M] and rewards [B, M, K].
    """
    b = batch["inputs"].shape[0]
    rep = {k: (v.repeat_interleave(members, 0) if torch.is_tensor(v) and v.ndim > 1 else v)
           for k, v in batch.items()}
    history = model.encode_history(rep)
    sampler = LatentDiffusionSampler(model.denoiser)
    pos = model.graphs.latent_pos
    steps, rewards = [], []
    for k in range(horizon):
        nf = model.node_forcing(rep["forcings_tgt"][:, k])
        zbar, _, _ = model.structural(history, nf)
        action = sampler.sample(model.graphs, zbar, history, nf, num_denoising_steps)
        z = zbar.exp(model._residual_scale(action), pos)
        rewards.append(-per_sample_error(model, model.decode(z), rep["targets"][:, k]))
        steps.append({"history": list(history), "zbar": zbar, "nf": nf, "action": action})
        history = list(history[1:]) + [z]
    r = torch.stack(rewards, 1)  # [B*M, K]
    discount = gamma ** torch.arange(horizon, device=r.device, dtype=r.dtype)
    returns = (r * discount).sum(1).view(b, members)
    return steps, returns, r.view(b, members, horizon)


def group_advantages(returns: torch.Tensor, beta: float):
    """GRPO-style normalised advantages [B, M] and regression weights (mean 1)."""
    g = returns.double()  # returns are close to each other: normalise in float64
    adv = (g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-6)
    weights = torch.softmax(adv / beta, dim=1) * returns.shape[1]
    return adv.to(returns.dtype), weights.to(returns.dtype)


def _rows(z: Latent, sl: slice) -> Latent:
    return Latent(z.h[sl], z.v[sl])


def awr_loss(model: ClimateWorldModel, steps: List[dict], weights: torch.Tensor):
    """Advantage-weighted EDM regression of the denoiser onto its own actions."""
    w = weights.reshape(-1)
    total = 0.0
    for s in steps:
        err, _ = model.denoiser.loss_per_sample(model.graphs, s["action"], s["zbar"],
                                                s["history"], s["nf"])
        total = total + (w * err).mean()
    return total / len(steps)


def awr_backward(model: ClimateWorldModel, steps: List[dict], weights: torch.Tensor,
                 scale: float = 1.0, chunk: Optional[int] = None) -> float:
    """Accumulates the gradient of scale * awr_loss one rollout step and `chunk`
    samples at a time, so only one chunk's graph is alive (the B*M*K graph of
    awr_loss does not fit next to the anchor on CPU). Returns awr_loss."""
    w = weights.reshape(-1)
    n = w.shape[0]
    chunk = chunk or n
    total = 0.0
    for s in steps:
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            err, _ = model.denoiser.loss_per_sample(
                model.graphs, _rows(s["action"], sl), _rows(s["zbar"], sl),
                [_rows(h, sl) for h in s["history"]], s["nf"][sl])
            part = (w[sl] * err).sum() / (n * len(steps))
            (scale * part).backward()
            total += float(part.detach())
    return total


def refine(model: ClimateWorldModel, loader: DataLoader, cfg: RefineConfig,
           device="cpu", log=print) -> List[Dict[str, float]]:
    if model.denoiser is None:
        raise ValueError("refinement needs a *_diffusion transition (the policy)")
    rl_weight = cfg.rl_weight if cfg.mode == "rl" else 0.0
    params = set_trainable(model, ("denoiser",))
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_lambda(step):
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * step / max(cfg.steps, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    batches = _forever(loader)
    history, t0 = [], time.time()
    for step in range(cfg.steps):
        batch = _to(next(batches), device)
        model.eval()  # sampling: no dropout / norm updates
        logs: Dict[str, float] = {}
        if rl_weight > 0:
            steps, returns, rewards = collect_rollouts(
                model, batch, cfg.members, cfg.horizon, cfg.gamma, cfg.num_denoising_steps)
            _, weights = group_advantages(returns, cfg.beta)
            logs.update(reward=float(rewards.mean()), return_spread=float(returns.std(1).mean()),
                        weight_max=float(weights.max()))
        model.train()
        opt.zero_grad(set_to_none=True)
        anchor, _ = model.losses(batch, "stochastic", {"gen": 1.0}, cfg.horizon)
        (cfg.anchor_weight * anchor).backward()
        logs["anchor"] = float(anchor.detach())
        loss = cfg.anchor_weight * logs["anchor"]
        if rl_weight > 0:
            awr = awr_backward(model, steps, weights, rl_weight,
                               cfg.awr_chunk or batch["inputs"].shape[0])
            loss += rl_weight * awr
            logs["awr"] = awr
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        logs.update(loss=loss, grad_norm=float(grad_norm), step=step)
        history.append(logs)
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            terms = " ".join(f"{k}={v:.4g}" for k, v in logs.items() if k not in ("step", "grad_norm"))
            log(f"[refine:{cfg.mode}] step {step:>4} {terms} ({time.time() - t0:.0f}s)")
    model.eval()
    return history
