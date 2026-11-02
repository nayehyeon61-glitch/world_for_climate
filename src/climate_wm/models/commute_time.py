"""Stage 8: commute-time embedding of atmospheric states (CTWM).

The transition cost C(X, Y) = inf_T [lambda T + inf_a int_0^T 1/2 |a|^2 dt]
(with dX/dt = F(X) + B(X) a) is a quasi-metric; its round trip
C(X, Y) + C(Y, X) has a stochastic counterpart, the commute time of the
transition graph. Hauri et al. 2026 (arXiv 2610.01373, "Learning
Commute-Time-Preserving World Models for Planning") learn an encoder phi and a
displacement predictor T with

    L = E |phi(s') - phi(s) - T(phi(s), a)|^2 + beta E |T(phi(s), a)|^2
        - gamma / 2 log det Cov phi(s)

For deterministic, reversible dynamics the optimum is the graph Laplacian's
eigenvectors scaled by lambda^-1/2: squared latent distances are commute
times and the variance of direction k is proportional to 1 / lambda_k.

Stage 8 phase 1 fits this as a head psi on fixed features of the frozen world
model (eval/topology.py). Variants:

    mode         global: one psi per state; patch: one psi per patch, shared
                 weights, the patch identity as input
    regulariser  logdet (CTWM) | vicreg (isotropic: variance hinge plus
                 off-diagonal covariance; the paper's ablation)
    pairs        next (s, s_t+1) | shuffle (s, random s'; the negative control)
    use_action   the forcing a_t (incoming solar radiation, season) as action
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor, nn


@dataclasses.dataclass
class CommuteConfig:
    dim: int = 16
    hidden: int = 128
    beta: float = 0.1             # slowness: cost of the predicted displacement
    gamma: float = 1.0            # weight of the spread term
    regulariser: str = "logdet"   # logdet | vicreg
    pairs: str = "next"           # next | shuffle
    use_action: bool = True
    mode: str = "global"          # global | patch
    steps: int = 3000
    batch_size: int = 256
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-4
    seed: int = 0


def _mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, o))


class CommuteTimeHead(nn.Module):
    """psi = phi(x) and the displacement T(psi, a); x is [N, D] (global) or [N, P, D] (patch)."""

    def __init__(self, in_dim: int, action_dim: int, cfg: CommuteConfig, num_patches: int = 0):
        super().__init__()
        self.cfg = cfg
        self.num_patches = num_patches
        self.action_dim = action_dim if cfg.use_action else 0
        self.phi = _mlp(in_dim + num_patches, cfg.hidden, cfg.dim)
        self.T = _mlp(cfg.dim + self.action_dim + num_patches, cfg.hidden, cfg.dim)

    def _ident(self, x: Tensor) -> Tensor:
        eye = torch.eye(self.num_patches, device=x.device, dtype=x.dtype)
        return eye.expand(x.shape[0], -1, -1)

    def embed(self, x: Tensor) -> Tensor:
        if self.num_patches:
            x = torch.cat([x, self._ident(x)], -1)
        return self.phi(x)

    def displacement(self, psi: Tensor, a: Optional[Tensor]) -> Tensor:
        parts = [psi]
        if self.action_dim:
            parts.append(a[:, None].expand(-1, psi.shape[1], -1) if psi.dim() == 3 else a)
        if self.num_patches:
            parts.append(self._ident(psi))
        return self.T(torch.cat(parts, -1))


def spread_penalty(psi: Tensor, cfg: CommuteConfig) -> Tensor:
    """-gamma/2 log det Cov psi (logdet) or the isotropic VICReg terms, per patch then averaged."""
    flat = psi if psi.dim() == 3 else psi[:, None]           # [B, P, k]
    x = flat - flat.mean(0, keepdim=True)
    cov = torch.einsum("bpi,bpj->pij", x, x) / max(x.shape[0] - 1, 1)  # [P, k, k]
    k = cov.shape[-1]
    eye = torch.eye(k, device=psi.device, dtype=psi.dtype)
    if cfg.regulariser == "logdet":
        return -0.5 * cfg.gamma * torch.logdet(cov + 1e-4 * eye).mean()
    if cfg.regulariser == "vicreg":
        std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1) + 1e-4)
        off = (cov * (1 - eye)).pow(2).sum((-2, -1)) / k
        return cfg.gamma * (torch.relu(1 - std).mean(-1) + off).mean()
    raise ValueError(f"unknown regulariser {cfg.regulariser!r}")


def commute_loss(head: CommuteTimeHead, x: Tensor, x_next: Tensor, a: Optional[Tensor]) -> Dict[str, Tensor]:
    cfg = head.cfg
    psi, psi_next = head.embed(x), head.embed(x_next)
    d = head.displacement(psi, a)
    pred = (psi_next - psi - d).pow(2).sum(-1).mean()
    slow = d.pow(2).sum(-1).mean()
    spread = spread_penalty(psi, cfg)
    return {"loss": pred + cfg.beta * slow + spread, "pred": pred, "slow": slow, "spread": spread}


def fit_head(x: np.ndarray, pairs: np.ndarray, action: Optional[np.ndarray], cfg: CommuteConfig,
             log=None) -> tuple:
    """Trains a head on states x [N, D] or [N, P, D] and transition pairs [M, 2] (i -> j).

    `action[i]` is the forcing of the transition that starts at i. Returns
    (head, history)."""
    torch.manual_seed(cfg.seed)
    gen = np.random.default_rng(cfg.seed)
    xt = torch.as_tensor(x, dtype=torch.float32)
    at = torch.as_tensor(action, dtype=torch.float32) if action is not None else None
    num_patches = x.shape[1] if (cfg.mode == "patch" and x.ndim == 3) else 0
    in_dim = x.shape[-1]
    head = CommuteTimeHead(in_dim, 0 if at is None else at.shape[1], cfg, num_patches)
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[dict] = []
    for step in range(cfg.steps):
        sel = pairs[gen.integers(0, len(pairs), cfg.batch_size)]
        i, j = sel[:, 0], sel[:, 1]
        if cfg.pairs == "shuffle":
            j = gen.integers(0, len(x), len(i))
        elif cfg.pairs != "next":
            raise ValueError(f"unknown pairs {cfg.pairs!r}")
        out = commute_loss(head, xt[i], xt[j], None if at is None else at[i])
        opt.zero_grad()
        out["loss"].backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % max(cfg.steps // 10, 1) == 0 or step == cfg.steps - 1:
            rec = {"step": step, **{k: float(v) for k, v in out.items()}}
            history.append(rec)
            if log:
                log(f"  step {step:5d}  " + "  ".join(f"{k} {v:.4f}" for k, v in rec.items() if k != "step"))
    head.eval()
    return head, history


@torch.no_grad()
def embed_all(head: CommuteTimeHead, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    xt = torch.as_tensor(x, dtype=torch.float32)
    return torch.cat([head.embed(xt[i:i + batch]) for i in range(0, len(xt), batch)]).numpy()
