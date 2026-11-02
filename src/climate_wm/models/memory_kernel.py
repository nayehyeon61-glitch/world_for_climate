"""Stage 10: an adaptive memory kernel for the physics world model.

Closing the equations on the resolved (or, in the game, observed) variables
X leaves, by the Mori-Zwanzig formalism, a memory integral of the unresolved
ones:

    dX/dt = F(X) + int_0^t K(tau, X) X(t - tau) dtau + noise.

With K a sum of exponentials, sum_k C_k exp(-tau / tau_k) B_k, the integral
is carried exactly by a linear recurrent state per grid column,

    m_k <- a_k m_k + (1 - a_k) b_k(x),   a_k = exp(-dt / tau_k),

read into the tendencies by C. "Adaptive": tau_k(x) and b_k(x) depend on the
column's state (a selective state space). At a re-observation the innovation
d = (observed - forecast) on the observed points is absorbed, m += G(x, d)
(the role of a Kalman gain), and the hidden points restart from the forecast
plus a readout R(m). The memory spreads horizontally by a learned diffusion
exp(-kappa_k n(n + 1)) of each channel.

All output layers start at zero, so an untrained memory changes nothing: the
model is then the physics model with nudging (direct insertion of the
observed points, keeping the forecast elsewhere) at re-observations.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

KNOCKOUTS = ("memory", "innovation", "readout", "spread", "adaptive", "long", "frozen")
# "frozen" (stage 15.5): m keeps its current value (no recurrence, no innovation) while the tendency
# network still reads it; with m = 0 it is the Markov version of the same column (the "memory" knock-out
# also drops the tendency network's m-independent part, which SPEEDY-adapted models rely on).


@dataclasses.dataclass
class MemoryConfig:
    channels: int = 16                 # K; 0 = Markov (no memory)
    adaptive: bool = True              # False: constant tau_k, linear B and G (the fixed kernel)
    tau_hours: Tuple[float, float] = (3.0, 240.0)  # initial timescales, log-spaced over the channels
    hidden: int = 64
    spread: bool = True                # learned horizontal diffusion of m
    innovation: bool = True            # absorb the innovation at re-observations (G)
    readout: bool = True               # hidden points restart from forecast + R(m)
    readout_max: Optional[float] = None  # stage 13 R1: soft bound |R(m)| <= readout_max (normalised units,
                                         # readout_max * tanh(R / readout_max)); the readout fed back into
                                         # a persistently hidden region blew the observed runs up (31/31)
    readout_decay_steps: Optional[float] = None  # stage 13 R2: a point hidden for a steps gets the readout
                                         # times exp(-(a - 1) / this) (the first hidden step gets it fully):
                                         # R was learned as a one-step correction, not an increment to integrate
    # stage 10.2: slow channels (months to years) with fixed timescales and a fixed input
    # projection b = tanh(W x) (W frozen): running means of the column state that only
    # the readers C and R learn to use. They take no innovation and no control.
    long_channels: int = 0
    long_tau_hours: Tuple[float, float] = (720.0, 17520.0)  # 1 month .. 2 years

    def __post_init__(self):
        self.tau_hours = tuple(self.tau_hours)
        self.long_tau_hours = tuple(self.long_tau_hours)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "MemoryConfig":
        return cls(**(d or {}))


def _net(d_in: int, hidden: int, d_out: int, zero: bool = True) -> nn.Sequential:
    net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, d_out))
    if zero:
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
    return net


class MemoryKernel(nn.Module):
    """Memory m [B, K, nlat, nlon] on the core's Gaussian grid.

    d_feat: column features (PhysicsWorldModel.column_features); n_out: the
    column network's tendency outputs it adds to; n_dyn: dynamic channels."""

    def __init__(self, cfg: MemoryConfig, d_feat: int, n_out: int, n_dyn: int, sh, call_seconds: float):
        super().__init__()
        self.cfg, self.sh_ref, self.dt = cfg, [sh], float(call_seconds)
        k, h, kl = cfg.channels, cfg.hidden, cfg.long_channels
        self.short, self.channels = k, k + kl  # the short (adaptive) channels come first
        tau0 = np.geomspace(cfg.tau_hours[0], cfg.tau_hours[1], k) * 3600.0
        if kl:
            tau0 = np.concatenate([tau0, np.geomspace(cfg.long_tau_hours[0], cfg.long_tau_hours[1], kl) * 3600.0])
            gen = torch.Generator().manual_seed(0)
            w = torch.randn(d_feat, kl, generator=gen) / np.sqrt(d_feat)
            self.register_buffer("inp_long", w)  # frozen projection of the column features
        self.register_buffer("log_tau0", torch.tensor(np.log(tau0), dtype=torch.float32))
        if cfg.adaptive:
            self.rate = _net(d_feat, h, k)             # log tau shift, zero at the start
            self.inp = _net(d_feat, h, k, zero=False)  # b(x)
            self.gain = _net(3 * n_dyn + k, h, k)      # G(forecast, innovation, observed, m)
        else:
            self.log_tau = nn.Parameter(torch.zeros(k))
            self.inp = nn.Linear(d_feat, k)
            self.gain = nn.Linear(2 * n_dyn, k)        # G(innovation, observed), linear
            nn.init.zeros_(self.gain.weight)
            nn.init.zeros_(self.gain.bias)
        self.tend = _net(d_feat + k + kl, h, n_out)    # C: memory term of the tendencies
        self.read = _net(k + kl, h, n_dyn)             # R: hidden-point restart correction
        self.kappa = nn.Parameter(torch.full((k + kl,), -3.0))  # softplus: diffusion per update at n = trunc
        self.knock: set = set()                        # evaluation knock-outs (KNOCKOUTS)
        self.control_tau: Optional[Tensor] = None      # [B, K, H, W] shift of ln tau set by a memory
                                                       # controller for the current step (stage 10 D-memctrl)
        self.track: Optional[list] = None              # set to a list: log10 tau (hours) per update,
                                                       # zonal means [H], quantiles 10/50/90 over all
                                                       # channels [3] and per channel [K * 3]

    # ---------------------------------------------------------------- kernel
    def tau(self, x: Tensor) -> Tensor:
        """Timescales [B, H, W, K] in seconds for column features x [B, H, W, F]
        (with a controller's shift `control_tau`, clipped to e^{+-4})."""
        k = self.short
        if not self.cfg.adaptive:
            log = (self.log_tau0[:k] + self.log_tau).expand(*x.shape[:-1], -1)
        elif "adaptive" in self.knock:
            log = self.log_tau0[:k].expand(*x.shape[:-1], -1)
        else:
            log = self.log_tau0[:k] + 2.0 * torch.tanh(self.rate(x))  # within e^{+-2} of tau0
        if self.control_tau is not None:
            log = log + self.control_tau.clamp(-4.0, 4.0).permute(0, 2, 3, 1)
        if self.channels > k:  # the slow channels keep their timescales
            log = torch.cat([log, self.log_tau0[k:].expand(*x.shape[:-1], -1)], -1)
        return torch.exp(log)

    def update(self, m: Tensor, x: Tensor) -> Tensor:
        """One step of the recurrence (one column-physics call)."""
        if "memory" in self.knock:
            return torch.zeros_like(m)
        if "frozen" in self.knock:
            return m
        tau = self.tau(x)
        if self.track is not None:
            lt = torch.log10(tau.detach().float() / 3600.0)  # [B, H, W, K]
            qs = torch.tensor([0.1, 0.5, 0.9], device=lt.device)
            flat = lt.reshape(-1, lt.shape[-1])
            flat = flat[:: max(1, flat.shape[0] // 20000)]
            # the per-channel spread is the state dependence (the channels start log-spaced)
            per = torch.quantile(flat, qs, dim=0).t().flatten()
            self.track.append(torch.cat([lt.mean((0, 2, 3)), torch.quantile(flat.flatten(), qs), per]).cpu())
        a = torch.exp(-self.dt / tau)
        b = torch.tanh(self.inp(x)) if self.cfg.adaptive else self.inp(x)
        if self.channels > self.short:
            b = torch.cat([b, torch.tanh(x @ self.inp_long)], -1)
        m = (a * m.permute(0, 2, 3, 1) + (1 - a) * b).permute(0, 3, 1, 2)
        return self.spread(m)

    def spread(self, m: Tensor) -> Tensor:
        if not self.cfg.spread or "spread" in self.knock:
            return m
        sh = self.sh_ref[0]
        lam = sh.eig / sh.eig.max()
        filt = torch.exp(-nn.functional.softplus(self.kappa).view(-1, 1, 1) * lam)  # [K, 1, N]
        return sh.to_grid(sh.to_spec(m) * filt)

    def _readable(self, m: Tensor) -> Tensor:
        """m with the slow channels zeroed under the "long" knock-out."""
        if "long" in self.knock and self.channels > self.short:
            m = torch.cat([m[:, :self.short], torch.zeros_like(m[:, self.short:])], 1)
        return m

    def tendency(self, x: Tensor, m: Tensor) -> Tensor:
        """Memory term [B, H, W, n_out] in the column network's output units."""
        if "memory" in self.knock:
            return x.new_zeros(*x.shape[:-1], self.tend[-1].out_features)
        return self.tend(torch.cat([x, self._readable(m).permute(0, 2, 3, 1)], -1))

    # ---------------------------------------------------------- observations
    def absorb(self, m: Tensor, forecast_g: Tensor, innov_g: Tensor, observed_g: Tensor) -> Tensor:
        """m after a re-observation; inputs [B, C_dyn, H, W] on the Gaussian grid."""
        if not self.cfg.innovation or self.knock & {"innovation", "memory", "frozen"}:
            return m
        k = self.short
        if self.cfg.adaptive:
            z = torch.cat([forecast_g, innov_g, observed_g, m[:, :k]], 1)
        else:
            z = torch.cat([innov_g, observed_g], 1)
        g = self.gain(z.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.channels > k:  # the slow channels take no innovation
            g = torch.cat([g, torch.zeros_like(m[:, k:])], 1)
        return m + g

    def readout(self, m: Tensor) -> Optional[Tensor]:
        """Correction [B, C_dyn, H, W] (normalised units, Gaussian grid) of the hidden points."""
        if not self.cfg.readout or self.knock & {"readout", "memory"}:
            return None
        r = self.read(self._readable(m).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if self.cfg.readout_max is not None:
            r = self.cfg.readout_max * torch.tanh(r / self.cfg.readout_max)
        return r
