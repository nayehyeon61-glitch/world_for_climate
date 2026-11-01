"""Stage 10.2: training the memory model as a continuous filter through the years.

The stage 10 game trains on 36 h episodes that start from an empty memory and
re-observe the truth with probability 0.5 per step. Its model lost the free
forecast (the 2020 cold start of 11/01: z500 at 24 h 601 against N1's 258
m^2/s^2, a blow-up by 120 h): it became a filter. Here the model runs through
the training years in order with the memory carried from window to window
(truncated backpropagation at the window boundaries). A window holds
`fill_steps` + `free_steps` six-hour steps: in the first `fill_steps` the full
truth is re-observed through the posterior after every step (the fill, as at
test time), the rest runs free. The loss is the GraphCast-weighted error of
every step's forecast, so the free segment trains the forecast from a filled
memory and the fill segment the one-step filter. The memory is carried from
the end of the free segment without a re-observation (the gain G then only
ever sees one-step innovations, as in the fill at test time).

Slow channels (MemoryConfig.long_channels: fixed timescales of months to years,
a frozen input projection) hold running means of the column state over the
whole pass; what is learned is how C and R read them. `expand_state_dict`
loads a stage 10 checkpoint into a model with slow channels added: the new
weights that read them start at zero, so the model starts as the checkpoint.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ..data.era5_lazy_loader import GraphCastWindowDataset
from .physics_game import _finite, attach_loss
from .prediction_game import sample_mask, errors, target_mask

STEP_S = 6 * 3600


@dataclasses.dataclass
class FilterConfig:
    fill_steps: int = 4        # re-observed steps per window (24 h)
    free_steps: int = 8        # free steps per window (48 h)
    epochs: int = 1
    lr_model: float = 1.0e-4   # column physics + surface head
    lr_new: float = 1.0e-3     # memory
    warmup_windows: int = 10
    grad_clip: float = 1.0
    cap: float = 10.0          # a window ends before a step whose error exceeds this
    train_model: bool = True   # False: only the memory learns
    new_prefix: str = "memory."  # the parameters at lr_new (stage 13: "corrector.")
    persistent_cap: float = 0.0  # stage 13 R3: a random region cap of this fraction stays hidden through the
                                 # whole window (forecast + readout there, the rest re-observed in the fill
                                 # steps), so the readout learns a filled memory and long-hidden points
    log_every: int = 10
    save_every: int = 10
    seed: int = 0

    @property
    def steps(self) -> int:
        return self.fill_steps + self.free_steps


def consecutive_windows(ds, layout, stats, start: str, end: str, steps: int, init_hours=None):
    """A window dataset of `steps` targets and the indices of the windows that
    tile [start, end) back to back (the start of each is the end of the last)."""
    data = GraphCastWindowDataset(ds, layout, stats, steps, start=start, end=end, init_hours=init_hours)
    starts = {int(data.times[i].astype("datetime64[s]").astype(np.int64)): j for j, i in enumerate(data.indices)}
    t0 = int(np.datetime64(start).astype("datetime64[s]").astype(np.int64))
    t1 = int(np.datetime64(end).astype("datetime64[s]").astype(np.int64))
    order, t = [], t0
    while t + steps * STEP_S <= t1 and t in starts:
        order.append(starts[t])
        t += steps * STEP_S
    return data, order


def expand_state_dict(model, state: Dict[str, Tensor]) -> List[str]:
    """Loads `state` into `model`, copying the leading slice of every tensor
    whose shape grew (slow channels added to the kernel: the new input columns
    of C and R, kappa and the timescales); returns the keys that were expanded."""
    own = model.state_dict()
    expanded, load = [], {}
    for key, value in state.items():
        if key in own and own[key].shape != value.shape:
            if value.ndim != own[key].ndim or any(a > b for a, b in zip(value.shape, own[key].shape)):
                raise ValueError(f"{key}: cannot place {tuple(value.shape)} into {tuple(own[key].shape)}")
            grown = own[key].clone()
            if key.endswith(".weight"):  # new input columns of a reader start at zero
                grown.zero_()
            grown[tuple(slice(0, n) for n in value.shape)] = value
            load[key] = grown
            expanded.append(key)
        else:
            load[key] = value
    missing = model.load_state_dict(load, strict=False).missing_keys
    new = ("memory.", "corrector.")  # modules a checkpoint may lack: they start fresh
    if any(not k.startswith(new) for k in missing):
        raise ValueError(f"checkpoint lacks {[k for k in missing if not k.startswith(new)]}")
    return expanded


def _schedule(cfg: FilterConfig, total: int):
    def f(step):
        if cfg.warmup_windows and step < cfg.warmup_windows:
            return (step + 1) / cfg.warmup_windows
        return 0.5 * (1 + math.cos(math.pi * min(step, total) / max(total, 1)))
    return f


def train_filter(model, dataset, order: List[int], cfg: FilterConfig, device="cpu",
                 progress: Optional[dict] = None, save: Optional[Callable[[dict], None]] = None,
                 log=None) -> Dict[str, list]:
    """One or more passes through the windows `order` of `dataset` in sequence
    (module docstring). `progress`: an earlier save to continue from."""
    attach_loss(model)
    model_params, new_params = [], []
    for name, p in model.named_parameters():
        if name.startswith(cfg.new_prefix):
            new_params.append(p)
        else:
            p.requires_grad_(cfg.train_model)
            if cfg.train_model:
                model_params.append(p)
    groups = [{"params": model_params, "lr": cfg.lr_model}, {"params": new_params, "lr": cfg.lr_new}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.0)
    total = cfg.epochs * len(order)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _schedule(cfg, total))
    history: Dict[str, list] = {"window": [], "loss": [], "fill": [], "free": [], "blowups": [], "grad": [],
                                "seconds": []}
    done, mem = 0, None
    if progress is not None:
        model.load_state_dict(progress["model"])
        opt.load_state_dict(progress["opt"])
        sched.load_state_dict(progress["sched"])
        done, mem, history = progress["done"], progress["mem"], progress["history"]
        if log is not None:
            log(f"[filter] resumed after {done}/{total} windows")
    t2d = model.layout.target_to_dynamic
    mask = None
    t_start = time.time()
    corrector = getattr(model, "corrector", None)
    soft = corrector is not None and corrector.cfg.soft_weight > 0  # stage 13 E3
    for n in range(done, total):
        j = order[n % len(order)]
        if n % len(order) == 0 and n > done:
            mem = None  # a new pass starts with an empty memory
        if soft:
            corrector.penalties = []
        item = dataset[j]
        batch = {k: v[None].to(device) for k, v in item.items()}
        if mask is None:
            mask = torch.zeros_like(batch["inputs"][:, -1])
            tmask = target_mask(model, mask)  # the loss counts every point, the cap included
        if cfg.persistent_cap > 0:  # R3: this window's hidden cap
            gen = torch.Generator().manual_seed(cfg.seed + 7919 * n)
            mask = sample_mask(model.layout, "region", dataset.lat, dataset.lon, gen, cfg.persistent_cap)[0][None].to(device)
        model.train(cfg.train_model)
        opt.zero_grad(set_to_none=True)
        state, ctx = model.begin(batch)
        if mem is None:
            snap = batch["inputs"][:, -1]
            state = model.posterior(state, ctx, snap, snap, mask)
        else:
            state["mem"] = mem.to(device)
        losses, fill_err, free_err, blown = [], [], [], 0
        for k in range(cfg.steps):
            state, precip = model.advance(state, ctx, k)
            pred = model.predict(state, ctx, k, precip)
            truth = batch["targets"][:, k]
            _, e = errors(model, pred, truth, tmask)
            if not (torch.isfinite(e).all() and _finite(state).all()) or float(e.detach()) > cfg.cap:
                blown = 1
                break
            losses.append(e.mean())
            (fill_err if k < cfg.fill_steps else free_err).append(float(e.detach()))
            if k < cfg.fill_steps:
                state = model.posterior(state, ctx, pred[:, t2d], truth[:, t2d], mask)
        gnorm = float("nan")
        if losses:
            loss = torch.stack(losses).mean()
            if soft and corrector.penalties:
                loss = loss + torch.stack(corrector.penalties).mean()
            if cfg.train_model or new_params:
                loss.backward()
                params = [p for g in opt.param_groups for p in g["params"]]
                gnorm = float(torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip))
                if math.isfinite(gnorm):
                    opt.step()
        sched.step()
        mem = state["mem"].detach() if "mem" in state and not blown and torch.isfinite(state["mem"]).all() else None
        model.eval()
        history["window"].append(n)
        history["loss"].append(float(np.mean(fill_err + free_err)) if losses else float("nan"))
        history["fill"].append(float(np.mean(fill_err)) if fill_err else float("nan"))
        history["free"].append(float(np.mean(free_err)) if free_err else float("nan"))
        history["blowups"].append(blown)
        history["grad"].append(gnorm)
        history["seconds"].append(time.time() - t_start)
        if log is not None and ((n + 1) % cfg.log_every == 0 or n + 1 == total):
            recent = slice(-cfg.log_every, None)
            log(f"[filter] {n + 1}/{total} fill {np.nanmean(history['fill'][recent]):.4f} "
                f"free {np.nanmean(history['free'][recent]):.4f} blow-ups {sum(history['blowups'][recent])} "
                f"grad {np.nanmean(history['grad'][recent]):.3f} ({time.time() - t_start:.0f} s)")
        if save is not None and ((n + 1) % cfg.save_every == 0 or n + 1 == total):
            save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "done": n + 1, "mem": None if mem is None else mem.cpu(), "history": history})
    for p in model.parameters():
        p.requires_grad_(True)
    return history
