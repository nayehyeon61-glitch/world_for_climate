"""Stage 7: prediction as a game, played by a controller on a frozen world model.

One episode = one initial state and a K-step (24 h) forecast. The game hides
part of the input (observational dropout, Freeman, Metz & Ha 2019):

    variables   one variable on every level (or the surface group)
    shell       every input channel of one shell
    region      a spherical cap covering `region_fraction` of the globe

and fills it with climatology (0 in normalised units). With `obs_prob` > 0
the (masked) truth is re-observed after a step with that chance, otherwise
the model continues from its own prediction. The controller
(models/feature_control.py) sees the state, its physical diagnostics and the
mask, and modulates the latent GNN features before every step. The score of
a step is the GraphCast-weighted MSE of the decoded forecast, on the hidden
points and on all points:

    R = - sum_k [ w * MSE_hidden_k + (1 - w) * MSE_all_k ]

The world model never changes (World Models: V and M frozen, only C learns).

    mode rl     M Gaussian action sequences per state; the action of step k
                is scored by its reward-to-go sum_{j>=k} r_j, normalised
                over the group (as in GRPO, train/rl_refine.py); policy
                gradient on the controller's mean action
    mode grad   the control: the same controller and game, trained by the
                exact gradient of R through the frozen model

Reanalysis does not react to the actions, so this is policy improvement from
verification reward, as in stage 2.
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.feature_control import FeatureController, controlled
from ..models.latent import Latent
from ..models.world_model import ClimateWorldModel
from .trainer import _forever, _to

MASK_KINDS = ("variables", "shell", "region")


@dataclasses.dataclass
class GameConfig:
    mode: str = "rl"              # rl | grad
    steps: int = 150
    batch_size: int = 2
    lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_steps: int = 10
    grad_clip: float = 1.0
    members: int = 4              # GRPO group size (rl)
    explore: float = 0.5          # std of the Gaussian policy (rl)
    horizon: int = 4              # 6 h steps per episode (a sequence x_t -> x_t+1 -> ...)
    mask_kinds: Tuple[str, ...] = MASK_KINDS
    mask_rate: float = 1.0        # share of episodes with a mask (0: the no-mask game)
    region_fraction: float = 0.25
    obs_prob: float = 0.0         # chance to re-observe the masked truth after a step
    nudge: Optional[float] = None  # re-observation: None = restart from the observed part
                                   # (hidden points at their mean), x = nudging of strength x
    masked_weight: float = 0.5
    log_every: int = 10
    seed: int = 0

    def __post_init__(self):
        self.mask_kinds = tuple(self.mask_kinds)
        if self.mode not in ("rl", "grad"):
            raise ValueError("game mode must be rl or grad")
        unknown = set(self.mask_kinds) - set(MASK_KINDS)
        if unknown:
            raise ValueError(f"unknown mask kinds {sorted(unknown)}")


# ------------------------------------------------------------------- masks
def variable_groups(layout) -> Dict[str, List[int]]:
    """Dynamic input channels per atmospheric variable, plus one surface group."""
    groups: Dict[str, List[int]] = {}
    for i, c in enumerate(layout.dynamic):
        groups.setdefault(c.name if c.level is not None else "surface", []).append(i)
    return groups


def sample_mask(layout, kind: str, lat: np.ndarray, lon: np.ndarray, gen: torch.Generator,
                region_fraction: float = 0.25) -> Tuple[Tensor, str]:
    """One episode's mask [C_dyn, H, W] (1 = hidden) and its label."""
    c, h, w = len(layout.dynamic), len(lat), len(lon)
    mask = torch.zeros(c, h, w)
    pick = lambda n: int(torch.randint(n, (1,), generator=gen))
    if kind == "variables":
        groups = variable_groups(layout)
        name = sorted(groups)[pick(len(groups))]
        mask[groups[name]] = 1
        return mask, f"variables:{name}"
    if kind == "shell":
        s = pick(len(layout.shells))
        part = layout.dynamic_shells[s]
        mask[part.scalar + part.vector_u + part.vector_v] = 1
        return mask, f"shell:{layout.shells.shells[s].name}"
    if kind == "region":
        centre = torch.randn(3, generator=gen)
        centre = centre / centre.norm()
        la, lo = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None]
        xyz = torch.tensor(np.stack(np.broadcast_arrays(np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo),
                                                        np.sin(la)), -1), dtype=torch.float32)
        cap = (xyz @ centre) > 1 - 2 * region_fraction  # cap area = (1 - cos theta) / 2
        mask[:, cap] = 1
        return mask, "region"
    raise ValueError(f"unknown mask kind {kind}")


def episode_masks(layout, batch_size: int, cfg: GameConfig, lat, lon,
                  gen: torch.Generator) -> Tuple[Tensor, List[str]]:
    masks, labels = [], []
    for _ in range(batch_size):
        if not cfg.mask_kinds or float(torch.rand(1, generator=gen)) >= cfg.mask_rate:
            masks.append(torch.zeros(len(layout.dynamic), len(lat), len(lon)))
            labels.append("none")
            continue
        kind = cfg.mask_kinds[int(torch.randint(len(cfg.mask_kinds), (1,), generator=gen))]
        m, label = sample_mask(layout, kind, lat, lon, gen, cfg.region_fraction)
        masks.append(m)
        labels.append(label)
    return torch.stack(masks), labels


def target_mask(model: ClimateWorldModel, mask: Tensor) -> Tensor:
    """Input mask [B, C_dyn, H, W] -> target mask [B, C_tgt, H, W]."""
    out = mask.new_zeros(mask.shape[0], len(model.layout.targets), *mask.shape[2:])
    out[:, model.layout.target_to_dynamic] = mask
    return out


# ------------------------------------------------------------------- score
def errors(model: ClimateWorldModel, pred: Tensor, target: Tensor, tmask: Tensor) -> Tuple[Tensor, Tensor]:
    """GraphCast-weighted MSE per sample on the hidden points and on all points [B]."""
    lf = model.loss_fn
    w = lf.channel_w[None, :, None, None] * lf.lat_w[None, None, :, None]  # [1, C, H, 1]
    err = (pred.float() - target.float()).square() * w
    all_ = err.sum((1, 2, 3)) / w.expand_as(err[:1]).sum()
    hidden_w = (w * tmask).sum((1, 2, 3))
    hidden = (err * tmask).sum((1, 2, 3)) / hidden_w.clamp_min(1e-12)
    return torch.where(hidden_w > 0, hidden, all_), all_


def _repeat(batch: Dict[str, Tensor], n: int) -> Dict[str, Tensor]:
    return {k: (v.repeat_interleave(n, 0) if torch.is_tensor(v) and v.ndim > 1 else v)
            for k, v in batch.items()}


def reobserved_fields(truth: Tensor, forecast: Tensor, mask: Tensor, nudge: Optional[float] = None) -> Tensor:
    """Fields a re-observation restarts from (normalised dynamic channels).
    The game's rule (nudge None): the observed part, hidden points at their
    mean (0). Nudging, the classical baseline: the forecast, with the observed
    points pulled towards the truth by `nudge` (1 = direct insertion)."""
    if nudge is None:
        return truth * (1 - mask)
    return forecast + nudge * (1 - mask) * (truth - forecast)


def play(model: ClimateWorldModel, ctrl: FeatureController, batch: Dict[str, Tensor],
         masks: Tensor, horizon: int, members: int = 1, control: bool = True,
         obs_prob: float = 0.0, gen: Optional[torch.Generator] = None,
         record: Optional[list] = None, nudge: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    """Plays one episode per initial state (members share the mask and the
    re-observation draws). Returns hidden and all-point errors [B * M, K].
    `nudge`: re-observe by nudging (see `reobserved_fields`)."""
    b = batch["inputs"].shape[0]
    rep = _repeat(batch, members)
    mask = masks.to(rep["inputs"].device).repeat_interleave(members, 0)
    tmask = target_mask(model, mask)
    obs = dict(rep, inputs=rep["inputs"] * (1 - mask[:, None]))
    t2d = model.layout.target_to_dynamic
    ctrl.mask, ctrl.enabled, ctrl.record, ctrl.decoded = mask, control, record, None
    hidden, all_ = [], []
    try:
        with controlled(model, ctrl):
            history = model.encode_history(obs)
            for k in range(horizon):
                z = model.step(history, rep["forcings_tgt"][:, k], sample=False).z
                pred = model.decode(z)
                e_h, e_a = errors(model, pred, rep["targets"][:, k], tmask)
                hidden.append(e_h)
                all_.append(e_a)
                ctrl.decoded = pred.detach()  # the controller's view of z (saves a decode)
                if obs_prob > 0 and k < horizon - 1:
                    seen = (torch.rand(b, generator=gen) < obs_prob).repeat_interleave(members)
                    if seen.any():
                        seen = seen.to(pred.device)
                        dyn = reobserved_fields(rep["targets"][:, k][:, t2d], pred[:, t2d], mask, nudge)
                        z_obs = model.encode(dyn, rep["forcings_tgt"][:, k], rep["static"])
                        sel = lambda a, o: torch.where(seen.view(-1, *[1] * (a.ndim - 1)), o, a)
                        z = Latent(sel(z.h, z_obs.h), sel(z.v, z_obs.v))
                        ctrl.decoded = None
                history = list(history[1:]) + [z]
    finally:
        ctrl.mask, ctrl.record, ctrl.decoded, ctrl.enabled = None, None, None, True
    return torch.stack(hidden, 1), torch.stack(all_, 1)


def step_rewards(hidden: Tensor, all_: Tensor, masked_weight: float) -> Tensor:
    """r_k per episode and step [N, K] from per-step errors [N, K]."""
    return -(masked_weight * hidden + (1 - masked_weight) * all_)


def episode_return(hidden: Tensor, all_: Tensor, masked_weight: float) -> Tensor:
    """R per episode [N] from per-step errors [N, K]."""
    return step_rewards(hidden, all_, masked_weight).sum(1)


def reward_to_go_advantages(rewards: Tensor, members: int) -> Tensor:
    """Group-normalised reward-to-go per step [N, K] (N = B * members): the
    action of step k only changes the rewards from step k on."""
    togo = rewards.double().flip(1).cumsum(1).flip(1)
    g = togo.view(-1, members, togo.shape[1])
    adv = (g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-6)
    return adv.view_as(togo).float()


# ---------------------------------------------------------------- training
def train_controller(model: ClimateWorldModel, ctrl: FeatureController, loader: DataLoader,
                     cfg: GameConfig, device="cpu", log=print, progress: Optional[dict] = None,
                     save: Optional[Callable[[dict], None]] = None, save_every: int = 10,
                     play_fn: Optional[Callable] = None) -> List[Dict[str, float]]:
    """Trains the controller; `save` receives a resumable `progress` dict every
    `save_every` updates (the same windows, masks and draws as an uninterrupted run).
    `play_fn` replaces `play` for another world model (stage 9: train/physics_game.py)."""
    play = play_fn or globals()["play"]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    params = [p for p in ctrl.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def lr_lambda(step):
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * step / max(cfg.steps, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    gen = torch.Generator().manual_seed(cfg.seed)
    batches = _forever(loader)
    history, t0 = [], time.time()
    if progress is not None:
        ctrl.load_state_dict(progress["controller"])
        opt.load_state_dict(progress["optimizer"])
        sched.load_state_dict(progress["scheduler"])
        gen.set_state(progress["generator"])
        history = list(progress["history"])
        for _ in range(len(history)):  # the loader's order is seeded: skip the windows already played
            next(batches)
        log(f"[game:{cfg.mode}] resumed at update {len(history)}")
    for step in range(len(history), cfg.steps):
        batch = _to(next(batches), device)
        b = batch["inputs"].shape[0]
        masks, labels = episode_masks(model.layout, b, cfg, loader.dataset.lat, loader.dataset.lon, gen)
        obs_seed = int(torch.randint(2 ** 31, (1,), generator=gen))
        obs_gen = lambda: torch.Generator().manual_seed(obs_seed)  # the same draws in every run
        logged = cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1)
        r0 = None
        if cfg.mode == "grad" or logged:  # the uncontrolled score of the same episodes
            with torch.no_grad():
                r0 = episode_return(*play(model, ctrl, batch, masks, cfg.horizon, control=False,
                                          obs_prob=cfg.obs_prob, gen=obs_gen(), nudge=cfg.nudge),
                                    cfg.masked_weight)
        opt.zero_grad(set_to_none=True)
        logs: Dict[str, float] = {}
        if cfg.mode == "rl":
            record: list = []
            ctrl.explore = cfg.explore
            with torch.no_grad():
                rewards = step_rewards(*play(model, ctrl, batch, masks, cfg.horizon, cfg.members,
                                             obs_prob=cfg.obs_prob, gen=obs_gen(), record=record, nudge=cfg.nudge),
                                       cfg.masked_weight)  # [B * M, K]
            ret = rewards.sum(1).view(b, cfg.members)
            adv = reward_to_go_advantages(rewards, cfg.members)
            loss = -sum((adv[:, k] * ctrl.log_prob(f, a)).mean() for k, (f, a) in enumerate(record)) / len(record)
            loss.backward()
            logs["return_spread"] = float(ret.std(1).mean() / ret.mean().abs())
            if r0 is not None:
                logs["gain"] = float(((ret.mean(1) - r0) / r0.abs()).mean())
        else:
            ctrl.explore = 0.0
            ret = episode_return(*play(model, ctrl, batch, masks, cfg.horizon,
                                       obs_prob=cfg.obs_prob, gen=obs_gen(), nudge=cfg.nudge), cfg.masked_weight)
            loss = -(ret / r0.abs()).mean()  # relative to the uncontrolled score of the episode
            loss.backward()
            logs["gain"] = float(((ret.detach() - r0) / r0.abs()).mean())
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()
        logs.update(loss=float(loss.detach()), grad_norm=float(grad_norm), step=step,
                    masks=sum(l != "none" for l in labels))
        if logged:
            ctrl.explore = 0.0
            with torch.no_grad():  # the mean policy on the same episodes
                rm = episode_return(*play(model, ctrl, batch, masks, cfg.horizon,
                                          obs_prob=cfg.obs_prob, gen=obs_gen(), nudge=cfg.nudge), cfg.masked_weight)
            logs["gain_mean_policy"] = float(((rm - r0) / r0.abs()).mean())
            terms = " ".join(f"{k}={v:.4g}" for k, v in logs.items() if k not in ("step",))
            log(f"[game:{cfg.mode}] step {step:>4} {terms} ({time.time() - t0:.0f}s) {labels}")
        history.append(logs)
        if save is not None and ((step + 1) % save_every == 0 or step == cfg.steps - 1):
            save({"controller": ctrl.state_dict(), "optimizer": opt.state_dict(),
                  "scheduler": sched.state_dict(), "generator": gen.get_state(), "history": history})
    ctrl.explore = 0.0
    return history


# -------------------------------------------------------------- evaluation
# baselines without a controller: re-observation by nudging, with this strength
NUDGE_VARIANTS = {"nudging": 1.0, "nudging_half": 0.5}
CONTROL_VARIANTS = ("no_control", "random_control", "no_diag", *NUDGE_VARIANTS)


@torch.no_grad()
def evaluate_game(model: ClimateWorldModel, ctrl: FeatureController, dataset, indices: Sequence[int],
                  kinds: Sequence[str], horizon: int, device="cpu", seed: int = 0,
                  region_fraction: float = 0.25, variants: Sequence[str] = CONTROL_VARIANTS,
                  obs_prob: float = 0.0, log=None, play_fn: Optional[Callable] = None,
                  nudge: Optional[float] = None) -> Dict[str, object]:
    """Masked-game errors per kind and lead for the controller, its knock-outs
    and the nudging baselines (NUDGE_VARIANTS, no controller), paired over
    initialisations (each init gets one fixed mask per kind and, with
    `obs_prob` > 0, one fixed sequence of re-observations). `nudge`: the
    re-observation rule the controller was trained with (GameConfig.nudge),
    used by it and its knock-outs; the baselines keep their own."""
    play = play_fn or globals()["play"]
    model.eval()
    ctrl.explore = 0.0
    out: Dict[str, object] = {"lead_h": [6 * (k + 1) for k in range(horizon)], "obs_prob": obs_prob,
                              "kinds": {}}
    for kind in kinds:
        rows: Dict[str, List[Tensor]] = {}
        labels = []
        for n, i in enumerate(indices):
            item = dataset[int(i)]
            batch = {k: v[None].to(device) for k, v in item.items()}
            gen = torch.Generator().manual_seed(seed + 1000 * n)
            if kind == "none":
                mask, label = torch.zeros(len(model.layout.dynamic), len(dataset.lat), len(dataset.lon)), "none"
            else:
                mask, label = sample_mask(model.layout, kind, dataset.lat, dataset.lon, gen, region_fraction)
            labels.append(label)
            for name in ("control", *variants):
                ctrl.random_actions = name == "random_control"
                ctrl.zero_diagnostics = name == "no_diag"
                torch.manual_seed(seed + n)
                h, a = play(model, ctrl, batch, mask[None], horizon,
                            control=name != "no_control" and name not in NUDGE_VARIANTS,
                            obs_prob=obs_prob, gen=torch.Generator().manual_seed(seed + 7919 * n),
                            nudge=NUDGE_VARIANTS.get(name, nudge))
                rows.setdefault(f"{name}_hidden", []).append(h[0])
                rows.setdefault(f"{name}_all", []).append(a[0])
            ctrl.random_actions = ctrl.zero_diagnostics = False
            if log is not None and (n + 1) % 10 == 0:
                log(f"[game eval] {kind}: {n + 1}/{len(indices)}")
        per = {k: torch.stack(v).cpu().numpy() for k, v in rows.items()}  # [N, K]
        res = {"labels": labels, "mse": {k: v.mean(0).tolist() for k, v in per.items()}, "change": {}}
        base = {part: per[f"no_control_{part}"] for part in ("hidden", "all")}
        for name in ("control", *[v for v in variants if v != "no_control"]):
            for part in ("hidden", "all"):
                rel = (per[f"{name}_{part}"] - base[part]) / base[part].mean(0)
                rng = np.random.default_rng(seed)
                idx = rng.integers(0, len(rel), (2000, len(rel)))
                means = rel[idx].mean(1)
                res["change"][f"{name}_{part}"] = {"mean": rel.mean(0).tolist(),
                                                   "lo": np.percentile(means, 5, 0).tolist(),
                                                   "hi": np.percentile(means, 95, 0).tolist()}
        out["kinds"][kind] = res
    return out
