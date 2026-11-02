"""Reality critic of stage 5: does a prediction look like the observed atmosphere?

The world model judges its own states with two critics whose reference
statistics are tracked (EMA, no gradient) on observed frames during training:

  DiagnosticCritic  moment matching in the space of the squashed physical
                    diagnostics (`eval.physics_diag`). Per channel and latitude
                    row a state has a zonal mean m and a zonal variance v; the
                    score compares them with the observed means,
                        (m - mu)^2 / v_obs  +  (log v - log v_obs)^2,
                    averaged over channels and rows (the mean is measured in
                    observed zonal standard deviations, which keeps channels
                    with an almost constant zonal mean from dominating). The
                    variance term makes a blurred prediction score badly, which
                    a plain likelihood would reward.
  LieFisherCritic   Fisher's linear discriminant on so(3) (Li et al., Lie Group
                    Machine Learning, ch. 9) between the generators implied by
                    the observed wind (Killing projection of the shell-mean
                    wind, rad per step, through the same pyramid projection as
                    the model) and the model's generators, per (shell, patch
                    level), in the patch's Cartan frame (spin about the normal,
                    east, north) with a pooled within-class covariance.
                    separability J = dmu^T S^-1 dmu (0 = indistinguishable);
                    the loss is the whitened distance of the batch mean of the
                    model generators to the observed mean.

Uses: an auxiliary loss ramped in over the dynamics stage (arm G3), member
weights softmax(-score / tau) (scores standardised over the members) for the
DIAMOND ensemble, and a reward term for the stage 2 refinement.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from ..eval.physics_diag import PhysicsDiagnostics


@dataclasses.dataclass
class CriticConfig:
    """Stage 5 critic options (model.critic in the config)."""
    enabled: bool = False
    fisher: bool = True  # Lie-Fisher part (pyramid transitions)
    momentum: float = 0.99  # EMA of the observed statistics
    ramp_steps: int = 200  # loss weight ramps 0 -> 1 over the first training steps
    tau: float = 1.0  # temperature of the member weights
    eps: float = 1e-4


def _ema(buf: Tensor, value: Tensor, momentum: float, first: bool) -> None:
    buf.copy_(value if first else momentum * buf + (1 - momentum) * value)


class DiagnosticCritic(nn.Module):

    def __init__(self, diag: PhysicsDiagnostics, num_lat: int, cfg: CriticConfig):
        super().__init__()
        self.diag = diag
        self.cfg = cfg
        k = len(diag)
        self.register_buffer("mean", torch.zeros(k, num_lat))  # E[zonal mean]
        self.register_buffer("log_var", torch.zeros(k, num_lat))  # E[log zonal variance]
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def moments(self, x: Tensor):
        """Normalised dynamic fields [B, C_dyn, H, W] -> zonal mean and log variance [B, K, H]."""
        f = self.diag(x)
        m = f.mean(-1)
        lv = torch.log(f.var(-1, unbiased=False) + self.cfg.eps)
        return m, lv

    @torch.no_grad()
    def observe(self, x: Tensor) -> None:
        m, lv = self.moments(x)
        first = bool(self.count == 0)
        mom = self.cfg.momentum
        _ema(self.mean, m.mean(0), mom, first)
        _ema(self.log_var, lv.mean(0), mom, first)
        self.count += 1

    def score(self, x: Tensor, per_channel: bool = False) -> Tensor:
        """[B] (or [B, K]) distance of each state's diagnostics to the observed ones."""
        m, lv = self.moments(x)
        d = (m - self.mean).square() / self.log_var.exp() + (lv - self.log_var).square()
        d = d.mean(-1)  # over latitude rows
        return d if per_channel else d.mean(-1)


class LieFisherCritic(nn.Module):

    def __init__(self, num_shells: int, centres: Sequence[Tensor], cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.num_levels = len(centres)
        n = torch.cat(list(centres), 0)  # [P, 3] patch normals, all levels
        z = torch.tensor([0., 0., 1.]).expand_as(n)
        east = torch.cross(z, n, dim=-1)
        east = torch.where(east.norm(dim=-1, keepdim=True) < 1e-6,
                           torch.tensor([1., 0., 0.]).expand_as(n), east)
        east = east / east.norm(dim=-1, keepdim=True)
        north = torch.cross(n, east, dim=-1)
        self.register_buffer("frame", torch.stack([n, east, north], 1))  # [P, 3, 3]
        level = torch.cat([torch.full((len(c),), i, dtype=torch.long) for i, c in enumerate(centres)])
        self.register_buffer("level", level)
        s, l = num_shells, self.num_levels
        self.register_buffer("mean", torch.zeros(2, s, l, 3))  # class 0 observed, 1 model
        self.register_buffer("second", torch.eye(3).repeat(2, s, l, 1, 1))  # E[f f^T]
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def features(self, omega: Tensor) -> Tensor:
        """Generators [B, S, P, 3] -> Cartan-frame components (spin, east, north)."""
        return torch.einsum("pij,bspj->bspi", self.frame, omega.float())

    def _level_moments(self, f: Tensor):
        """[B, S, P, 3] -> per-level mean [S, L, 3] and second moment [S, L, 3, 3]."""
        onehot = nn.functional.one_hot(self.level, self.num_levels).to(f.dtype)  # [P, L]
        w = onehot / onehot.sum(0, keepdim=True)
        mean = torch.einsum("bspi,pl->sli", f, w) / f.shape[0]
        second = torch.einsum("bspi,bspj,pl->slij", f, f, w) / f.shape[0]
        return mean, second

    @torch.no_grad()
    def observe(self, observed: Tensor, model: Tensor) -> None:
        first = bool(self.count == 0)
        for c, omega in enumerate((observed, model)):
            m, s2 = self._level_moments(self.features(omega))
            _ema(self.mean[c], m, self.cfg.momentum, first)
            _ema(self.second[c], s2, self.cfg.momentum, first)
        self.count += 1

    def within_cov(self) -> Tensor:
        """Pooled within-class covariance [S, L, 3, 3] (regularised)."""
        cov = self.second - self.mean[..., :, None] * self.mean[..., None, :]
        eye = torch.eye(3, device=cov.device)
        pooled = 0.5 * (cov[0] + cov[1])
        scale = pooled.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None].clamp_min(1e-12)
        return pooled + self.cfg.eps * scale * eye

    def separability(self) -> Tensor:
        """J = dmu^T S^-1 dmu per (shell, level) [S, L]."""
        d = (self.mean[0] - self.mean[1])[..., None]
        return (d.transpose(-1, -2) @ torch.linalg.solve(self.within_cov(), d))[..., 0, 0]

    def penalty(self, model: Tensor) -> Tensor:
        """Whitened squared distance of the batch-mean model generator to the observed mean."""
        m, _ = self._level_moments(self.features(model))
        d = (m - self.mean[0])[..., None]
        cov = self.within_cov().detach()
        return (d.transpose(-1, -2) @ torch.linalg.solve(cov, d))[..., 0, 0].mean()

    def score(self, model: Tensor) -> Tensor:
        """[B] Fisher discriminant per state: positive = on the observed side."""
        cov = self.within_cov()
        w = torch.linalg.solve(cov, (self.mean[0] - self.mean[1])[..., None])[..., 0]  # [S, L, 3]
        mid = 0.5 * (self.mean[0] + self.mean[1])
        f = self.features(model)
        lv = self.level
        proj = ((f - mid[:, lv]) * w[:, lv]).sum(-1)  # [B, S, P]
        return proj.mean((1, 2))


class RealityCritic(nn.Module):
    """Both critics with the loss ramp (model.critic)."""

    def __init__(self, diag: PhysicsDiagnostics, num_lat: int, num_shells: int,
                 centres: Optional[Sequence[Tensor]], cfg: CriticConfig):
        super().__init__()
        self.cfg = cfg
        self.diagnostic = DiagnosticCritic(diag, num_lat, cfg)
        self.fisher = LieFisherCritic(num_shells, centres, cfg) if (cfg.fisher and centres) else None
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def ramp(self) -> float:
        return min(1.0, float(self.steps) / max(self.cfg.ramp_steps, 1))

    def tick(self) -> None:
        self.steps += 1

    def summary(self) -> Dict[str, float]:
        out = {}
        if self.fisher is not None and int(self.fisher.count) > 0:
            out["fisher_J"] = float(self.fisher.separability().mean())
        return out


def member_weights(scores: Tensor, tau: float = 1.0, standardize: bool = True) -> Tensor:
    """Ensemble member weights softmax(-score / tau) over members (dim 0).

    With `standardize` the scores are first centred and scaled by their spread
    over the members, so that tau is a dimensionless sharpness (the raw score
    differences between members of one forecast are small and vary by lead).
    """
    if standardize and scores.shape[0] > 1:
        scores = (scores - scores.mean(0)) / (scores.std(0) + 1e-8)
    return torch.softmax(-scores / tau, 0)
