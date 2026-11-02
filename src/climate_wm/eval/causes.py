"""Cause analysis of the forecast error: ablations that need no retraining.

Error decomposition. At every lead k the forecast error of the decoded
rollout splits exactly into what the autoencoder cannot represent and what
the latent dynamics get wrong:

    total = D(Z_k) - X_k
          = [D(E(X_k)) - X_k]  +  [D(Z_k) - D(E(X_k))]
               rec (floor)          dyn (rollout in latent space)

with Z_k the rolled-out latent and E(X_k) the encoded truth. The RMSE of
`rec` is the best any transition could reach with this autoencoder, so a
model that loses at short leads with rec close to total has an autoencoder
problem, and one whose dyn grows fastest has a dynamics problem.

Eval-time ablations. The same rollout (same initial states, same denoiser
noise) with one part switched off:

    no_denoiser   the stochastic residual skipped (Zbar only: the pure group
                  action plus the composite terms), i.e. how much skill the
                  transport itself carries
    composite     every fibre family and the vertical shift set to the
                  identity (stage 5 arms), i.e. the arm without its new terms
    shift, radiation, mixing, surface, condensation
                  one part of the composite group set to the identity
    nonlocal      the stage 6 shell graphs of the GNN processors restricted
                  to the chain edges (s-1, s+1): do the non-local shell links
                  carry skill?
    shell_graph   the stage 6 coupling of the shells' generators removed
                  (independent shells)
    no_control    the stage 7 feature controller detached (the frozen model)
    random_control
                  the controller's actions applied at shuffled (shell, patch)
                  places: is the gain from steering or from any perturbation?
    no_diag       the controller without its physical-diagnostics input

Each variant is compared with the full model per initialisation, so the
reported change carries a paired bootstrap interval.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..geometry.composite_group import FIB_FAMILIES
from ..models.layers import ShellGraphMixer
from ..models.world_model import ClimateWorldModel
from ..train.losses import normalized_latitude_weights
from .graphcast_metrics import rmse_per_init

COMPOSITE_PARTS = ("composite", "shift") + FIB_FAMILIES
GRAPH_PARTS = ("nonlocal", "shell_graph")
CONTROL_PARTS = ("no_control", "random_control", "no_diag")


def _graph_mixers(model: ClimateWorldModel) -> List[ShellGraphMixer]:
    return [m for m in model.modules() if isinstance(m, ShellGraphMixer)]


def available_variants(model: ClimateWorldModel) -> List[str]:
    """Ablations that apply to this model."""
    out = ["no_denoiser"] if model.denoiser is not None else []
    if _graph_mixers(model):
        out.append("nonlocal")
    if getattr(model.lie, "shell_graph", None) is not None:
        out.append("shell_graph")
    if getattr(model, "feature_control", None) is not None:
        out += list(CONTROL_PARTS)
    lie = model.lie
    if lie is not None and getattr(lie, "composite", None) is not None:
        out.append("composite")
        if lie.shift is not None:
            out.append("shift")
        out += [f for f in FIB_FAMILIES if f in lie.composite.families]
    return out


@contextlib.contextmanager
def knocked_out(model: ClimateWorldModel, parts: Sequence[str]) -> Iterator[None]:
    """Sets parts of the composite group to the identity, or restricts the
    stage 6 shell graphs (GRAPH_PARTS), or switches the stage 7 controller
    (CONTROL_PARTS); restores everything on exit.

    A family's coefficients are scaled by its gain (`fibre_gain`), so a zero
    gain removes the family exactly; a zeroed shift head gives w = 0.
    """
    control = set(parts) & set(CONTROL_PARTS)
    if control:
        with _control_knocked_out(model, control), knocked_out(model, set(parts) - control):
            yield
        return
    graph = set(parts) & set(GRAPH_PARTS)
    parts = set(parts) & set(COMPOSITE_PARTS)
    if graph:
        with _graph_knocked_out(model, graph), knocked_out(model, parts):
            yield
        return
    if not parts:
        yield
        return
    lie = model.lie
    if lie is None or getattr(lie, "composite", None) is None:
        raise ValueError(f"knock-out {sorted(parts)} needs a composite transition")
    if "composite" in parts:
        parts |= {"shift", *FIB_FAMILIES}
    saved = {"gain": lie.fibre_gain.detach().clone()}
    if lie.shift is not None:
        saved["shift"] = [p.detach().clone() for p in lie.shift.parameters()]
    with torch.no_grad():
        for f in parts & set(FIB_FAMILIES):
            lie.fibre_gain[FIB_FAMILIES.index(f)] = 0.0
        if "shift" in parts and lie.shift is not None:
            for p in lie.shift.parameters():
                p.zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            lie.fibre_gain.copy_(saved["gain"])
            if lie.shift is not None:
                for p, v in zip(lie.shift.parameters(), saved["shift"]):
                    p.copy_(v)


@contextlib.contextmanager
def _graph_knocked_out(model: ClimateWorldModel, parts: set) -> Iterator[None]:
    mixers = _graph_mixers(model) if "nonlocal" in parts else []
    coupling = getattr(model.lie, "shell_graph", None) if "shell_graph" in parts else None
    if ("nonlocal" in parts and not mixers) or ("shell_graph" in parts and coupling is None):
        raise ValueError(f"knock-out {sorted(parts)} needs the stage 6 shell graph")
    saved = coupling.detach().clone() if coupling is not None else None
    for m in mixers:
        m.local_only = True
    if coupling is not None:
        with torch.no_grad():
            coupling.zero_()
    try:
        yield
    finally:
        for m in mixers:
            m.local_only = False
        if coupling is not None:
            with torch.no_grad():
                coupling.copy_(saved)


@contextlib.contextmanager
def _control_knocked_out(model: ClimateWorldModel, parts: set) -> Iterator[None]:
    ctrl = getattr(model, "feature_control", None)
    if ctrl is None:
        raise ValueError(f"knock-out {sorted(parts)} needs an attached feature controller")
    saved = ctrl.enabled, ctrl.random_actions, ctrl.zero_diagnostics
    ctrl.enabled = "no_control" not in parts
    ctrl.random_actions = "random_control" in parts
    ctrl.zero_diagnostics = "no_diag" in parts
    try:
        yield
    finally:
        ctrl.enabled, ctrl.random_actions, ctrl.zero_diagnostics = saved


def shell_graph_summary(model: ClimateWorldModel,
                        traces: Optional[Dict[str, list]] = None) -> Dict[str, object]:
    """The learned shell graphs: per mixer the bias-only weights softmax(B) and,
    from `traces`, the mean state-dependent weights with the share of weight on
    non-chain edges; the generator coupling A."""
    out: Dict[str, object] = {}
    names = {id(m): n for n, m in model.named_modules()}
    mixers = {}
    for m in _graph_mixers(model):
        nonlocal_share = lambda w: float((w * (~m.chain_edge).cpu().float()).sum(-1).mean())
        with torch.no_grad():
            prior = torch.softmax(m.bias.masked_fill(m.self_edge, float("-inf")), -1).cpu()
        entry = {"prior": prior.tolist(), "prior_nonlocal_share": nonlocal_share(prior)}
        trace = (traces or {}).get(names[id(m)])
        if trace:
            mean = torch.stack(trace).mean(0)
            entry.update(mean=mean.tolist(), nonlocal_share=nonlocal_share(mean))
        mixers[names[id(m)]] = entry
    if mixers:
        out["mixers"] = mixers
    coupling = getattr(model.lie, "shell_graph", None)
    if coupling is not None:
        out["generator_coupling"] = coupling.detach().cpu().tolist()
    return out


def latent_rollout(model: ClimateWorldModel, batch: Dict[str, Tensor], steps: int,
                   sample: bool = True) -> List:
    """Rolled-out latents Z_1 .. Z_steps (one member)."""
    history = model.encode_history(batch)
    out = []
    for k in range(steps):
        z = model.step(history, batch["forcings_tgt"][:, k], sample=sample).z
        out.append(z)
        history = list(history[1:]) + [z]
    return out


def _batches(dataset, indices: Sequence[int], batch_size: int, device):
    indices = list(indices)
    for i in range(0, len(indices), batch_size):
        chunk = [dataset[int(j)] for j in indices[i:i + batch_size]]
        yield {k: torch.utils.data.default_collate([c[k] for c in chunk]).to(device)
               for k in chunk[0]}


def _bootstrap(diff: np.ndarray, n: int = 2000, seed: int = 0) -> tuple:
    """Mean and 90% interval of the mean of `diff` over its first axis."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    means = diff[idx].mean(1)
    return diff.mean(0), np.percentile(means, 5, axis=0), np.percentile(means, 95, axis=0)


@torch.no_grad()
def error_decomposition(model: ClimateWorldModel, dataset, indices: Sequence[int], stats,
                        lead_steps: int, variants: Sequence[str] = (), device="cpu",
                        batch_size: int = 2, seed: int = 0, sample: bool = True,
                        log: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Per-variable, per-lead RMSE of total / rec / dyn and of each ablated variant
    (`sample` False: the denoiser is off in every run, the stage 7 game setting)."""
    model.eval()
    layout = model.layout
    lat_w = torch.tensor(normalized_latitude_weights(dataset.lat), dtype=torch.float32)
    phys = lambda x: stats.unnormalize(x.float().cpu(), layout.targets)
    rows: Dict[str, List[np.ndarray]] = {}
    add = lambda k, v: rows.setdefault(k, []).append(v)
    done = 0
    names = {id(m): n for n, m in model.named_modules()}
    traces = {names[id(m)]: [] for m in _graph_mixers(model)}
    for n_b, batch in enumerate(_batches(dataset, indices, batch_size, device)):
        target = phys(batch["targets"][:, :lead_steps])  # [B, L, C, H, W]
        rec = torch.stack([phys(model.decode(model.encode_target(batch, k)))
                           for k in range(lead_steps)], 1)
        runs = {}
        for name in ("full", *variants):
            torch.manual_seed(seed + n_b)  # the same denoiser noise for every variant
            for m in _graph_mixers(model):  # mean shell-graph weights of the full model
                m.trace = traces[names[id(m)]] if name == "full" else None
            with knocked_out(model, [name]):
                zs = latent_rollout(model, batch, lead_steps, sample=sample and name != "no_denoiser")
            runs[name] = torch.stack([phys(model.decode(z)) for z in zs], 1)
        for m in _graph_mixers(model):
            m.trace = None
        err = lambda a, b: rmse_per_init(a, b, lat_w).numpy()  # [B, L, C]
        add("rec", err(rec, target))
        add("dyn", err(runs["full"], rec))
        for name, pred in runs.items():
            add(name, err(pred, target))
        done += target.shape[0]
        if log is not None:
            log(f"[causes] {done}/{len(indices)} initialisations")
    per_init = {k: np.concatenate(v, 0) for k, v in rows.items()}  # [N, L, C]
    full = per_init["full"]
    out = {"lead_h": [6 * (k + 1) for k in range(lead_steps)],
           "variables": [c.key for c in layout.targets],
           "shells": [sh.name for sh in layout.shells.shells],
           "inits": len(full),
           "rmse": {k: v.mean(0).tolist() for k, v in per_init.items()}}
    # share of the squared error the autoencoder alone explains (mean over inits)
    out["rec_share"] = (np.square(per_init["rec"]).mean(0) / np.square(full).mean(0).clip(1e-30)).tolist()
    changes = {}
    for name in variants:
        rel = (per_init[name] - full) / full.mean(0).clip(1e-30)  # per init, relative to the mean
        mean, lo, hi = _bootstrap(rel, seed=seed)
        changes[name] = {"mean": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    out["variant_change"] = changes  # (variant - full) / full RMSE, paired over inits
    graphs = shell_graph_summary(model, traces)
    if graphs:
        out["shell_graph"] = graphs
    return out
