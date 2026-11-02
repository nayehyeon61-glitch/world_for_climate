"""Stage 16 (README_KO section 16): the tunable knobs of the physics world model (N1).

RAIN (Nath et al. 2024, arXiv 2408.16118) lets a reinforcement-learning policy set the
tunable constants of a climate model's physics from the model's state, the physics
itself unchanged. Here the knobs are multipliers of N1's column physics and two
constants of its dynamical core, set per sample (each episode of a batch has its own):

  Q_strat, Q_upper, Q_lower, Q_pbl   heating Q by level band (<= 100, 150-400, 500-850, >= 925 hPa)
  S_upper, S_lower, S_pbl            moistening S (<= 400, 500-850, >= 925 hPa)
  F                                  the column's momentum forcing (its column-mean-free part)
  drag                               the column's surface drag and the core's Rayleigh drag
  diffusion                          the hyperdiffusion e-folding time, tau = tau_0 * 2 ** a

An action a in [-1, 1]^10 maps to multipliers 1 + span * a and to tau = tau_0 * 2 ** (span * a);
a = 0 is the model as trained. Span presets: v1 = 0.5 for every multiplier (0.5 to 1.5) and 1 for the
diffusion (tau_0 / 2 to 2 tau_0); v2 narrows the moistening knobs, which act like a cliff (the
one-knob scan of 11/02: S_upper x 0.5 blows N1 up within a day, x 1.5 by day 5).
"""
from __future__ import annotations

import contextlib
from typing import Dict, Iterable, Optional, Sequence

import torch
from torch import Tensor

KNOBS = ("Q_strat", "Q_upper", "Q_lower", "Q_pbl", "S_upper", "S_lower", "S_pbl", "F", "drag", "diffusion")
GROUPS = {"heating": ("Q_strat", "Q_upper", "Q_lower", "Q_pbl"), "moistening": ("S_upper", "S_lower", "S_pbl"),
          "momentum": ("F", "drag"), "diffusion": ("diffusion",)}
SPAN = 0.5
SPAN_PRESETS = {
    "v1": (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0),
    "v2": (0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.5, 0.5, 1.0),
}


def span_of(spans) -> Tensor:
    """A preset name, a sequence of 10 spans, or None/() (= v1) -> [10]."""
    if not spans:
        spans = "v1"
    if isinstance(spans, str):
        spans = SPAN_PRESETS[spans]
    t = torch.tensor([float(x) for x in spans])
    assert t.shape == (len(KNOBS),), t.shape
    return t


def q_band(p: float) -> str:
    return "Q_strat" if p <= 100 else "Q_upper" if p <= 400 else "Q_lower" if p <= 850 else "Q_pbl"


def s_band(p: float) -> str:
    return "S_upper" if p <= 400 else "S_lower" if p <= 850 else "S_pbl"


def group_mask(groups: Iterable[str]) -> Tensor:
    """1 for the knobs in `groups` (names of GROUPS), 0 elsewhere [len(KNOBS)]."""
    on = {k for g in groups for k in GROUPS[g]}
    return torch.tensor([1.0 if k in on else 0.0 for k in KNOBS])


def settings(action: Tensor, levels: Sequence[float], spans=None) -> Dict[str, Tensor]:
    """Action [B, 10] in [-1, 1] -> per-sample multipliers: Q, S [B, L], F, drag, diffusion power [B]."""
    a = action.clamp(-1, 1) * span_of(spans).to(action.device, action.dtype)
    m = 1.0 + a
    idx = {k: i for i, k in enumerate(KNOBS)}
    q = torch.stack([m[:, idx[q_band(p)]] for p in levels], 1)
    s = torch.stack([m[:, idx[s_band(p)]] for p in levels], 1)
    return {"Q": q, "S": s, "F": m[:, idx["F"]], "drag": m[:, idx["drag"]],
            "diffusion_power": 2.0 ** (-a[:, idx["diffusion"]])}


def apply(model, action: Optional[Tensor], spans=None) -> None:
    """Sets the knobs of a PhysicsWorldModel (None clears them)."""
    if action is None:
        model.column_scale = None
        model.core.diffusion_power = None
        model.core.drag_scale = None
        return
    st = settings(action, model.levels, spans)
    model.column_scale = {k: st[k] for k in ("Q", "S", "F", "drag")}
    model.core.diffusion_power = st["diffusion_power"]
    model.core.drag_scale = st["drag"]


@contextlib.contextmanager
def knobs(model, action: Optional[Tensor], spans=None):
    apply(model, action, spans)
    try:
        yield model
    finally:
        apply(model, None)
