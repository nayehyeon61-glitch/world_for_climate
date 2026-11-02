"""Stage 9 arm N3: the stage 7 prediction game played on the physics world model.

Same masks, observational dropout, rewards and training loops as stage 7
(train/prediction_game.py: train_controller / evaluate_game with
`play_fn=play_physics`); the world model is the frozen core + column physics
and the controller (models/tendency_control.py) nudges the physical state in
low spherical harmonics before every 6 h step. While training, the control
cost `cost_weight * 1/2 |a|^2` is added to both error terms of a step, so
the step reward is -(w hidden + (1 - w) all) - cost.

Blow-ups: the learned column physics can blow up on masked states (found in
stage 10). With `cap` (GameConfig.cap), a sample whose forecast or state is
not finite scores `cap` from then on and restarts from its initial state, so
the batch, the rewards and the advantages stay finite.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from ..train.losses import GraphCastLoss
from .prediction_game import _repeat, errors, reobserved_fields, target_mask


def attach_loss(model) -> None:
    """The game scores with the GraphCast loss weights (not part of the model's state)."""
    if getattr(model, "loss_fn", None) is None:
        loss = GraphCastLoss(model.layout, model.lat).to(model.mean.device)
        object.__setattr__(model, "loss_fn", loss)


def _select(seen: Tensor, new: Tensor, old: Tensor) -> Tensor:
    return torch.where(seen.view(-1, *[1] * (old.ndim - 1)), new, old)


def _finite(state: Dict[str, Tensor]) -> Tensor:
    """Per-sample finiteness [B] of a core state."""
    ok = None
    for v in state.values():
        r = torch.view_as_real(v) if torch.is_complex(v) else v
        f = torch.isfinite(r).flatten(1).all(1)
        ok = f if ok is None else ok & f
    return ok


def _detach(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {k: v.detach() for k, v in state.items()}


def play_physics(model, ctrl, batch, masks: Tensor, horizon: int, members: int = 1,
                 control: bool = True, obs_prob: float = 0.0, gen: Optional[torch.Generator] = None,
                 record: Optional[list] = None, nudge: Optional[float] = None,
                 cap: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    """One episode per initial state (members share the mask and the
    re-observation draws). Returns hidden and all-point errors [B * M, K].
    `nudge`: re-observe by nudging (prediction_game.reobserved_fields).
    `cap`: errors are clipped there, and a blown-up sample scores `cap` from
    then on and restarts from its initial state (module docstring)."""
    attach_loss(model)
    b = batch["inputs"].shape[0]
    rep = _repeat(batch, members)
    rep["init_time"] = batch["init_time"].repeat_interleave(members)  # 1-d: not repeated by _repeat
    mask = masks.to(rep["inputs"].device).repeat_interleave(members, 0)
    tmask = target_mask(model, mask)
    obs = dict(rep, inputs=rep["inputs"] * (1 - mask[:, None]))
    t2d = model.layout.target_to_dynamic
    ctrl.enabled, ctrl.record = control, record
    ctrl.set_mask(mask)
    hidden, all_ = [], []
    try:
        state, ctx = model.begin(obs)
        if cap is not None:
            first, first_resid = _detach(state), ctx["resid"].detach()
            dead = torch.zeros(b * members, dtype=torch.bool, device=mask.device)
        for k in range(horizon):
            state = ctrl(state, k)
            cost = 0.0
            if ctrl.last_cost is not None and ctrl.cost_weight:
                cost = ctrl.cost_weight * ctrl.last_cost
            state, precip = model.advance(state, ctx, k)
            pred = model.predict(state, ctx, k, precip)
            e_h, e_a = errors(model, pred, rep["targets"][:, k], tmask)
            if cap is not None:
                bad = ~(torch.isfinite(e_h) & torch.isfinite(e_a) & _finite(state))
                dead = dead | bad
                e_h = torch.where(dead, torch.full_like(e_h, cap), e_h.clamp(max=cap))
                e_a = torch.where(dead, torch.full_like(e_a, cap), e_a.clamp(max=cap))
                if bad.any():  # restart blown-up samples so the batch stays finite
                    state = {key: _select(bad, first[key], v) for key, v in state.items()}
                    ctx["resid"] = _select(bad, first_resid, ctx["resid"])
                    pred = torch.nan_to_num(pred)
            hidden.append(e_h + cost)
            all_.append(e_a + cost)
            if obs_prob > 0 and k < horizon - 1:
                seen = (torch.rand(b, generator=gen) < obs_prob).repeat_interleave(members)
                if seen.any():
                    seen = seen.to(pred.device)
                    dyn = reobserved_fields(rep["targets"][:, k][:, t2d], pred[:, t2d], mask, nudge)
                    old_resid = ctx["resid"]
                    fresh = model.reobserve(ctx, dyn)
                    ctx["resid"] = _select(seen, ctx["resid"], old_resid)
                    state = {key: _select(seen, fresh[key], state[key]) for key in state}
    finally:
        ctrl.set_mask(None)
        ctrl.record, ctrl.enabled = None, True
    return torch.stack(hidden, 1), torch.stack(all_, 1)
