"""Stage 15 control F1: a pure neural-operator stepper (no physics core).

x_{k+1} = x_k + SFNO([x_k, x_{k-1}, forcings_{k+1}, static, action]) on the data grid,
in the normalised target space. It has the interface the stage 15 teacher loop uses
(begin / advance / predict, `track_tendency`, `last_tendency`), so N1-S and F1 train on
the same episodes with the same state loss and are scored by the same evaluation
(rule T3: the physics core stays stable longer than F1; rule T4 later reuses F1 with
memory as F2). There is no column, so `last_tendency` stays None (no tendency loss).
"""
from __future__ import annotations

import dataclasses
from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor, nn


@dataclasses.dataclass
class StepperConfig:
    hidden: int = 64
    layers: int = 4
    modes: int = 16          # SFNO modes (degree and order)
    action_channels: int = 3
    grid: str = "equiangular"

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "StepperConfig":
        d = dict(d or {})
        d.pop("type", None)
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})


def _sfno(n_in: int, n_out: int, cfg: StepperConfig) -> nn.Module:
    import functools
    try:
        from neuralop.layers.spherical_convolution import SphericalConv
        from neuralop.models import SFNO
    except ImportError as e:  # optional dependency (neuraloperator, torch-harmonics)
        raise ImportError("the F1 stepper needs the neuraloperator and torch-harmonics packages") from e
    conv = functools.partial(SphericalConv, sht_grids=cfg.grid)
    net = SFNO(n_modes=(cfg.modes, cfg.modes), in_channels=n_in, out_channels=n_out,
               hidden_channels=cfg.hidden, n_layers=cfg.layers, positional_embedding=None, conv_module=conv)
    last = net.projection.fcs[-1]
    nn.init.zeros_(last.weight)  # starts as persistence
    nn.init.zeros_(last.bias)
    return net


class NeuralStepper(nn.Module):
    """F1: residual SFNO stepper in the normalised target space."""

    def __init__(self, cfg: StepperConfig, layout, lat: np.ndarray, net: Optional[nn.Module] = None):
        super().__init__()
        self.cfg, self.layout = cfg, layout
        self.lat = np.asarray(lat)
        n_t, n_d = len(layout.targets), len(layout.dynamic)
        self.n_in = 2 * n_t + len(layout.forcings) + len(layout.static) + cfg.action_channels
        self.net = net if net is not None else _sfno(self.n_in, n_t, cfg)
        self.register_buffer("mean", torch.zeros(n_t))
        self.register_buffer("std", torch.ones(n_t))
        self.register_buffer("in_mean", torch.zeros(n_d))
        self.register_buffer("in_std", torch.ones(n_d))
        self.register_buffer("t2d", torch.tensor(layout.target_to_dynamic, dtype=torch.long), persistent=False)
        self.track_tendency = False
        self.last_tendency = None
        self.loss_fn = None

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        sd = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        for k in [k for k in sd if k.endswith("_metadata")]:  # neuralop metadata is not a tensor
            sd.pop(k)
        return sd

    def set_normalization(self, stats) -> None:
        get = lambda keys: torch.tensor([stats.mean[k] for k in keys], dtype=torch.float32)
        gsd = lambda keys: torch.tensor([stats.std[k] for k in keys], dtype=torch.float32)
        t = [c.key for c in self.layout.targets]
        d = [c.key for c in self.layout.dynamic]
        self.mean.copy_(get(t)); self.std.copy_(gsd(t))
        self.in_mean.copy_(get(d)); self.in_std.copy_(gsd(d))

    def _to_target(self, x_dyn: Tensor) -> Tensor:
        """Normalised dynamic inputs [B, C_dyn, H, W] -> normalised target layout (missing channels 0)."""
        phys = x_dyn * self.in_std[:, None, None] + self.in_mean[:, None, None]
        out = x_dyn.new_zeros((x_dyn.shape[0], len(self.layout.targets)) + x_dyn.shape[-2:])
        out[:, self.t2d] = (phys - self.mean[self.t2d, None, None]) / self.std[self.t2d, None, None]
        return out

    def begin(self, batch: Dict[str, Tensor]):
        x = self._to_target(batch["inputs"][:, -1])
        prev = self._to_target(batch["inputs"][:, -2]) if batch["inputs"].shape[1] > 1 else x
        ctx = {"static": batch["static"], "forcings_tgt": batch.get("forcings_tgt"),
               "action": batch.get("action")}
        return {"x": x, "prev": prev}, ctx

    def advance(self, state, ctx, k: int):
        x, prev = state["x"], state["prev"]
        parts = [x, prev]
        if ctx["forcings_tgt"] is not None and len(self.layout.forcings):
            parts.append(ctx["forcings_tgt"][:, k].to(x.dtype))
        parts.append(ctx["static"].to(x.dtype))
        if self.cfg.action_channels:
            a = ctx.get("action")
            parts.append(a.to(x.dtype) if a is not None else
                         x.new_zeros((x.shape[0], self.cfg.action_channels) + x.shape[-2:]))
        dx = self.net(torch.cat(parts, 1))
        return {"x": x + dx, "prev": x}, None

    def predict(self, state, ctx, k: int, precip_d=None) -> Tensor:
        return state["x"]

    def forward_rollout(self, batch: Dict[str, Tensor], steps: int) -> Tensor:
        state, ctx = self.begin(batch)
        preds = []
        for k in range(steps):
            state, _ = self.advance(state, ctx, k)
            preds.append(self.predict(state, ctx, k))
        return torch.stack(preds, 1)

    def rollout(self, batch: Dict[str, Tensor], steps: int, members: int = 1, **_) -> Tensor:
        """Normalised predictions [M, B, L, C, H, W] (deterministic: members are copies), as the
        physics model's, for pipeline.evaluate / compare_graphcast."""
        pred = self.forward_rollout(batch, steps)
        return pred[None].expand(members, *pred.shape)
