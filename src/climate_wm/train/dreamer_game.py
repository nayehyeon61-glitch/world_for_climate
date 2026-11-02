"""Stage 10: DreamerV3-style fluid training of the physics world model with memory.

The stage 7/9 game (masks, observational dropout, GraphCast-weighted hidden
and all-point errors) with two changes:

1. The world model keeps learning. Every iteration it plays masked 2018-19
   sequences with the current actor in the loop and is updated by the
   prediction loss (column physics, surface head, memory kernel), as the
   world model of DreamerV3 is trained on replayed experience.
2. The agent has memory. At a re-observation the model's posterior keeps its
   own forecast on the hidden points plus the memory's readout, and the
   memory absorbs the innovation (PhysicsWorldModel.posterior); without
   memory the rule is nudging with strength 1.

The actor (the stage 9 tendency controller, also reading the memory's low
harmonics) and a critic learn on episodes imagined in the current world model
(no gradient into it), `members` sampled actions per start:

    snapshot (default, 감자 19:09 KST)  the episode starts from a masked
        snapshot of one time (memory 0), exactly as a test episode, and the
        game's ERA5 re-observations (same mask, probability obs_prob) arrive
        during it; the actions cannot change the atmosphere, so its future
        observations are known for the game years, and the reward is the
        exact (negative) error, the episode ending at the horizon;
    blind  DreamerV3's imagination: from a posterior state of the real
        rollout, `imagine_horizon` steps without observations, scored by a
        reward head trained on the real rollouts.

lambda-returns, the return scale S = EMA(P95 - P5) and the REINFORCE loss
-sg((R - v) / max(1, S)) log pi follow DreamerV3; the critic regresses
symlog returns towards its own EMA copy. Departures: symlog regression
instead of twohot heads, a 6-step horizon, a fixed policy std (so the
entropy bonus is constant), and the reanalysis itself as the replay data
(actions cannot change the atmosphere, only the model's trajectory).
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..models.tendency_control import TendencyControlConfig, TendencyController
from .physics_game import _detach, _finite, _select, attach_loss
from .prediction_game import (MASK_KINDS, GameConfig, _repeat, episode_masks, errors, reobserved_fields,
                              sample_mask, step_rewards, target_mask)
from .trainer import _forever, _to

Rule = Union[str, float, None]  # "posterior" (stage 10), "reset" (the stage 7/9 game) or a nudging strength


@dataclasses.dataclass
class DreamerConfig:
    steps: int = 100
    batch_size: int = 2
    horizon: int = 6                # real rollout (6 h steps)
    imagine_horizon: int = 6
    members: int = 4                # actions sampled per imagined start state
    explore: float = 0.5            # policy std (action units)
    gamma: float = 0.95
    lam: float = 0.95
    lr_model: float = 1e-4          # column physics and surface head
    lr_new: float = 1e-3            # memory, reward head, actor, critic
    warmup_steps: int = 10
    grad_clip: float = 1.0
    train_model: bool = True        # False: frozen world model (arm D-frozen)
    train_actor: bool = True        # False: no actor (arm D-noactor)
    critic_ema: float = 0.98
    critic_reg: float = 1.0
    imagination: str = "snapshot"   # snapshot: episodes from a masked snapshot with ERA5 re-observations
                                    # and exact rewards; blind: DreamerV3's (posterior start, no
                                    # observations, reward head)
    cap: float = 10.0               # snapshot: a blown-up episode scores this error from then on
    blowup_penalty: float = 10.0    # blind: per remaining imagined step when the state blows up
    return_ema: float = 0.99
    advantage: str = "scale"        # scale: DreamerV3's (R - v) / max(1, S); group: GRPO-style, each step's
                                    # return against the members of the same start (as stages 7, 9 and 11)
    head_hidden: int = 128
    mask_kinds: Tuple[str, ...] = MASK_KINDS
    mask_rate: float = 1.0
    region_fraction: float = 0.25
    obs_prob: float = 0.5
    masked_weight: float = 0.5
    log_every: int = 5
    seed: int = 0

    def __post_init__(self):
        self.mask_kinds = tuple(self.mask_kinds)
        if self.imagination not in ("snapshot", "blind"):
            raise ValueError("imagination must be snapshot or blind")
        if self.advantage not in ("scale", "group"):
            raise ValueError("advantage must be scale or group")

    def game(self) -> GameConfig:
        return GameConfig(mask_kinds=self.mask_kinds, mask_rate=self.mask_rate,
                          region_fraction=self.region_fraction, obs_prob=self.obs_prob)


def symlog(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: Tensor) -> Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


# ------------------------------------------------------------------ agent
class MemoryController(TendencyController):
    """The stage 9 tendency controller, also reading the latest masked observation
    (low harmonics of the observed fields per variable group, and its age in
    steps / 6) and the low harmonics of the memory.

    `cfg.target = "memory"` (arm D-memctrl, 감자: adaptive memory through
    control): the action does not touch the physical state. Per memory channel
    it writes low harmonics (n <= cfg.mem_degree) into m and shifts ln tau for
    the next 6 h, so the controller decides what the memory keeps and for how
    long, and the physics sees it only through the memory's tendency term and
    readout."""

    def __init__(self, model, cfg: TendencyControlConfig, diff_std=None):
        super().__init__(model, cfg, diff_std)
        self.mem_channels = model.memory.cfg.channels if model.memory is not None else 0
        self.obs_dim = self.n_coef * len(self.mask_index) + 1
        self.in_dim += self.obs_dim + self.mem_channels * self.n_coef
        self.net[0] = nn.Linear(self.in_dim, cfg.hidden)
        self.last_obs: Optional[Tensor] = None  # [N, obs_dim], set by the game
        if cfg.target == "memory":
            if not self.mem_channels:
                raise ValueError("control.target=memory needs a memory (memory.channels > 0)")
            n = cfg.mem_degree
            self.mem_coords = [(m, l, part) for m in range(n + 1) for l in range(m, n + 1)
                               for part in ((0,) if m == 0 else (0, 1))]
            self.n_action = 2 * self.mem_channels * len(self.mem_coords)  # write and ln tau per channel
            self.net[-1] = nn.Linear(cfg.hidden, self.n_action)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def apply(self, state: Dict[str, Tensor], action: Tensor) -> Dict[str, Tensor]:
        if self.cfg.target != "memory":
            return super().apply(state, action)
        memory = self.model_ref[0].memory
        sh = memory.sh_ref[0]
        a = action.view(action.shape[0], 2, self.mem_channels, len(self.mem_coords))
        like = sh.to_spec(state["mem"])
        grid = lambda x, unit: sh.to_grid(self._from_low(x * unit, like, self.mem_coords))
        memory.control_tau = grid(a[:, 1], self.cfg.mem_tau_unit)  # held for this step's updates
        write = grid(a[:, 0], self.cfg.mem_write_unit)
        if write.shape[1] < state["mem"].shape[1]:  # the slow channels (10.2) take no control
            write = torch.cat([write, torch.zeros_like(state["mem"][:, write.shape[1]:])], 1)
        return dict(state, mem=state["mem"] + write)

    def observation_features(self, obs: Tensor, mask: Tensor) -> Tensor:
        """Masked observation (normalised dynamic fields [N, C_dyn, H, W], 1 = hidden) -> [N, obs_dim] (age 0)."""
        sh = self.model_ref[0].core.sh
        seen = obs * (1 - mask)
        per = torch.stack([seen[:, idx].mean(1) if idx else torch.zeros_like(seen[:, 0])
                           for idx in self.mask_index], 1)
        low = self._low(sh.from_data(per)).flatten(1)
        return torch.cat([low, low.new_zeros(low.shape[0], 1)], 1)

    def age(self) -> None:
        """One step passes since the latest observation."""
        if self.last_obs is not None:
            self.last_obs = torch.cat([self.last_obs[:, :-1], self.last_obs[:, -1:] + 1.0 / 6.0], 1)

    def features(self, state: Dict[str, Tensor], k: int) -> Tensor:
        x = super().features(state, k)
        b = x.shape[0]
        obs = self.last_obs if self.last_obs is not None else x.new_zeros(b, self.obs_dim)
        parts = [x, obs]
        if self.mem_channels:
            sh = self.model_ref[0].core.sh
            mem = self._low(sh.to_spec(state["mem"][:, :self.mem_channels])).flatten(1)
            if not self.cfg.use_state or self.zero_diagnostics:
                mem = torch.zeros_like(mem)
            parts.append(mem)
        return torch.cat(parts, 1)


class Heads(nn.Module):
    """Critic v(s) and reward head r(s, a), both in symlog units, zero-initialised."""

    def __init__(self, in_dim: int, n_action: int, hidden: int):
        super().__init__()

        def net(d):
            n = nn.Sequential(nn.Linear(d, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                              nn.Linear(hidden, 1))
            nn.init.zeros_(n[-1].weight)
            nn.init.zeros_(n[-1].bias)
            return n

        self.critic = net(in_dim)
        self.reward = net(in_dim + n_action)
        self.critic_ema = copy.deepcopy(self.critic)
        for p in self.critic_ema.parameters():
            p.requires_grad_(False)

    def value(self, f: Tensor, ema: bool = False) -> Tensor:
        return symexp((self.critic_ema if ema else self.critic)(f)[..., 0])

    def predicted_reward(self, f: Tensor, a: Tensor) -> Tensor:
        return symexp(self.reward(torch.cat([f, a], -1))[..., 0])

    @torch.no_grad()
    def update_ema(self, decay: float) -> None:
        for e, p in zip(self.critic_ema.parameters(), self.critic.parameters()):
            e.mul_(decay).add_(p, alpha=1 - decay)


# ------------------------------------------------------------------- game
def play_memory(model, ctrl: Optional[MemoryController], batch, masks: Tensor, horizon: int,
                members: int = 1, control: bool = True, explore: float = 0.0, obs_prob: float = 0.0,
                gen: Optional[torch.Generator] = None, rule: Rule = "posterior",
                record: Optional[list] = None, keep: Optional[list] = None,
                noise: Optional[torch.Generator] = None, cap: Optional[float] = None,
                truncate: bool = False, truncate_above: Optional[float] = None,
                obs_steps: Optional[int] = None,
                on_step: Optional[Callable[[int, Tensor, Tensor], None]] = None) -> Tuple[Tensor, Tensor]:
    """One masked episode per initial state; hidden and all-point errors [B * M, K]
    (differentiable through the world model when grad is enabled; the actor's
    actions are not). `rule`: re-observation (see Rule). `record` receives
    (features, action, cost) per step, `keep` the detached posterior states
    (state, resid, k, ctx, latest observation features) the imagination starts from.

    Blow-ups (a non-finite forecast; the learned column physics can blow up on
    masked states): `truncate` stops the episode before the first one (the
    errors have K' <= K steps; with gradients, play one sample at a time, as a
    NaN anywhere in a batched graph poisons the parameter gradients;
    `truncate_above` also stops at a finite all-point error above it, since a
    blown-up state teaches nothing), `cap` (without gradients) bounds every
    step's score, control cost included, scores a blown-up sample `cap` from
    then on and restarts its state from the initial one so the batch stays
    finite.

    `obs_steps`: re-observations only after the first `obs_steps` steps (a
    spin-up, then a free run; eval/memory_climate.py). `on_step(k, pred, dead)`
    receives every step's forecast and the samples blown up so far."""
    attach_loss(model)
    b = batch["inputs"].shape[0]
    rep = _repeat(batch, members)
    rep["init_time"] = batch["init_time"].repeat_interleave(members)
    mask = masks.to(rep["inputs"].device).repeat_interleave(members, 0)
    tmask = target_mask(model, mask)
    obs = dict(rep, inputs=rep["inputs"] * (1 - mask[:, None]))
    t2d = model.layout.target_to_dynamic
    acting = ctrl is not None and control
    if acting:
        ctrl.set_mask(mask)
    if model.memory is not None:
        model.memory.control_tau = None
    hidden, all_ = [], []
    try:
        state, ctx = model.begin(obs)
        snap = obs["inputs"][:, -1]
        if rule == "posterior" and model.memory is not None:
            # the memory takes in the masked snapshot; the hidden points start from its readout
            state = model.posterior(state, ctx, snap, snap, mask)
        if acting:
            ctrl.last_obs = ctrl.observation_features(snap, mask)
        first, first_resid = _detach(state), ctx["resid"].detach()
        dead = torch.zeros(b * members, dtype=torch.bool, device=mask.device)
        for k in range(horizon):
            if keep is not None:
                keep.append((_detach(state), ctx["resid"].detach(), k, ctx,
                             ctrl.last_obs if acting else None))
            cost = 0.0
            if acting:
                with torch.no_grad():
                    f = ctrl.features(state, k)
                    a = ctrl.net(f)
                    if explore > 0:
                        a = a + explore * torch.randn(a.shape, generator=noise).to(a.device)
                c = 0.5 * a.square().sum(1)
                cost = ctrl.cost_weight * c
                if record is not None:
                    record.append((f, a, c))
                state = ctrl.apply(state, a)
                ctrl.age()
            state, precip = model.advance(state, ctx, k)
            pred = model.predict(state, ctx, k, precip)
            e_h, e_a = errors(model, pred, rep["targets"][:, k], tmask)
            bad = ~(torch.isfinite(e_h) & torch.isfinite(e_a) & _finite(state))
            if truncate_above is not None:
                bad = bad | (e_a > truncate_above)
            if truncate and bad.any():
                if record is not None and acting:
                    record.pop()
                break
            e_h, e_a = e_h + cost, e_a + cost
            if cap is not None:
                dead = dead | bad
                # the cap bounds the whole score: the control cost of a huge action (a state
                # on its way to a blow-up) once overflowed to inf here and made the critic NaN
                e_h = torch.where(dead, torch.full_like(e_h, cap), torch.nan_to_num(e_h, nan=cap).clamp(max=cap))
                e_a = torch.where(dead, torch.full_like(e_a, cap), torch.nan_to_num(e_a, nan=cap).clamp(max=cap))
                if bad.any():  # restart blown-up samples so the batch stays finite
                    state = {key: _select(bad, first[key], v) for key, v in state.items()}
                    ctx["resid"] = _select(bad, first_resid, ctx["resid"])
                    pred = torch.nan_to_num(pred)
            hidden.append(e_h)
            all_.append(e_a)
            if on_step is not None:
                on_step(k, pred, dead)
            if obs_prob > 0 and k < horizon - 1 and (obs_steps is None or k < obs_steps):
                seen = (torch.rand(b, generator=gen) < obs_prob).repeat_interleave(members)
                if seen.any():
                    seen = seen.to(pred.device)
                    truth, fc = rep["targets"][:, k][:, t2d], pred[:, t2d]
                    old_resid = ctx["resid"]
                    if rule == "posterior":
                        fresh = model.posterior(state, ctx, fc, truth, mask)
                    else:
                        fresh = model.reobserve(ctx, reobserved_fields(truth, fc, mask,
                                                                       None if rule == "reset" else rule))
                        if "mem" in state:
                            fresh["mem"] = state["mem"]
                    ctx["resid"] = _select(seen, ctx["resid"], old_resid)
                    state = {key: _select(seen, fresh[key], state[key]) for key in state}
                    if acting:
                        ctrl.last_obs = _select(seen, ctrl.observation_features(truth, mask), ctrl.last_obs)
    finally:
        if acting:
            ctrl.set_mask(None)
            ctrl.last_obs = None
        if model.memory is not None:
            model.memory.control_tau = None
    if not hidden:
        empty = mask.new_zeros(b * members, 0)
        return empty, empty
    return torch.stack(hidden, 1), torch.stack(all_, 1)


@torch.no_grad()
def imagine(model, ctrl: MemoryController, heads: Heads, start, mask: Tensor, horizon: int,
            members: int, explore: float, noise: Optional[torch.Generator] = None,
            blowup_penalty: float = 10.0):
    """Imagined rollout from a posterior state `start` (an entry of play_memory's
    `keep`; no observations, no truth): features [T + 1, N, F], actions
    [T, N, A], predicted rewards [T, N] and continuation flags [T, N]. A
    sample whose state blows up ends there (continuation 0) and is charged
    `blowup_penalty` per remaining step (rewards are negative errors, so an
    unpenalised end would pay); its state restarts from `start` to keep the
    batch finite, with zero features from then on."""
    state, resid, k0, ctx, last_obs = start
    rep = lambda x: x.repeat_interleave(members, 0)
    state = {k: rep(v) for k, v in state.items()}
    init_state = dict(state)
    ictx = dict(ctx, resid=rep(resid), static=rep(ctx["static"]),
                static_g=rep(ctx["static_g"]) if ctx["static_g"] is not None else None,
                init=np.repeat(ctx["init"], members), forcings_tgt=None)
    ctrl.set_mask(rep(mask))
    ctrl.last_obs = rep(last_obs) if last_obs is not None else None
    feats, acts, rews, conts = [], [], [], []
    alive = torch.ones(state["temp"].shape[0], dtype=torch.bool, device=state["temp"].device)
    try:
        for j in range(horizon):
            f = ctrl.features(state, k0 + j) * alive[:, None]
            a = ctrl.net(f) + explore * torch.randn(f.shape[0], ctrl.n_action, generator=noise).to(f.device)
            r = heads.predicted_reward(f, a) - ctrl.cost_weight * 0.5 * a.square().sum(1)
            state, _ = model.advance(ctrl.apply(state, a), ictx, k0 + j)
            ctrl.age()
            ok = _finite(state) & alive
            r = torch.where(alive & ~ok, r - blowup_penalty * (horizon - j), r)
            rews.append(torch.where(alive, r, torch.zeros_like(r)))
            conts.append(ok.float())
            feats.append(f)
            acts.append(a)
            if (alive & ~ok).any():
                state = {key: _select(~ok, init_state[key], v) for key, v in state.items()}
            alive = ok
        feats.append(ctrl.features(state, k0 + horizon) * alive[:, None])
    finally:
        ctrl.set_mask(None)
        ctrl.last_obs = None
        if model.memory is not None:
            model.memory.control_tau = None
    return torch.stack(feats), torch.stack(acts), torch.stack(rews), torch.stack(conts)


def lambda_returns(rewards: Tensor, values: Tensor, gamma: float, lam: float,
                   cont: Optional[Tensor] = None) -> Tensor:
    """R_t = r_t + gamma c_t ((1 - lam) v_{t+1} + lam R_{t+1}), R_T = v_T;
    rewards and continuation flags [T, N], values [T + 1, N]."""
    cont = torch.ones_like(rewards) if cont is None else cont
    out, nxt = [], values[-1]
    for t in reversed(range(rewards.shape[0])):
        nxt = rewards[t] + gamma * cont[t] * ((1 - lam) * values[t + 1] + lam * nxt)
        out.append(nxt)
    return torch.stack(out[::-1])


def group_advantage(ret: Tensor, alive: Tensor) -> Tensor:
    """GRPO-style advantage [T, M]: each step's return against the mean and std
    of the alive members of the same start (zero where fewer than two are alive).
    The return-scale normalisation of DreamerV3 divides by the spread of returns
    across starts, which the mask kind sets; with one start and four members per
    iteration it left the action's own effect (~1% of that spread) invisible,
    and the trained actors were inert (11/01)."""
    n = alive.sum(1, keepdim=True)
    mean = (ret * alive).sum(1, keepdim=True) / n.clamp(min=1)
    var = (((ret - mean) * alive) ** 2).sum(1, keepdim=True) / (n - 1).clamp(min=1)
    adv = (ret - mean) / (var.sqrt() + 1e-6)
    return torch.where((n >= 2) & (alive > 0), adv, torch.zeros_like(adv))


# --------------------------------------------------------------- training
def _schedule(cfg: DreamerConfig):
    def f(step):
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * step / max(cfg.steps, 1)))
    return f


def train_dreamer(model, ctrl: Optional[MemoryController], heads: Optional[Heads], loader: DataLoader,
                  cfg: DreamerConfig, device="cpu", log=print, progress: Optional[dict] = None,
                  save: Optional[Callable[[dict], None]] = None, save_every: int = 5) -> List[Dict[str, float]]:
    """Alternates world-model and actor-critic updates (module docstring).
    `save` receives a resumable progress dict every `save_every` iterations."""
    attach_loss(model)
    acting = cfg.train_actor and ctrl is not None
    model_params, new_params = [], []
    for name, p in model.named_parameters():
        p.requires_grad_(cfg.train_model)
        if cfg.train_model:
            (new_params if name.startswith("memory.") else model_params).append(p)
    if acting:
        new_params += list(heads.reward.parameters())
    groups = [{"params": model_params, "lr": cfg.lr_model}, {"params": new_params, "lr": cfg.lr_new}]
    opt_m = torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=0.0) if model_params or new_params else None
    opt_a = (torch.optim.AdamW(list(ctrl.parameters()) + list(heads.critic.parameters()), lr=cfg.lr_new)
             if acting else None)
    scheds = [torch.optim.lr_scheduler.LambdaLR(o, _schedule(cfg)) for o in (opt_m, opt_a) if o is not None]
    gen = torch.Generator().manual_seed(cfg.seed)
    noise = torch.Generator().manual_seed(cfg.seed + 1)
    batches = _forever(loader)
    history: List[Dict[str, float]] = []
    scale = None
    if progress is not None:
        model.load_state_dict(progress["model"])
        if acting:
            ctrl.load_state_dict(progress["controller"])
            heads.load_state_dict(progress["heads"])
            opt_a.load_state_dict(progress["opt_agent"])
        if opt_m is not None:
            opt_m.load_state_dict(progress["opt_model"])
        for s, st in zip(scheds, progress["schedulers"]):
            s.load_state_dict(st)
        gen.set_state(progress["generator"])
        noise.set_state(progress["noise"])
        history, scale = list(progress["history"]), progress["return_scale"]
        for _ in range(len(history)):
            next(batches)
        log(f"[dreamer] resumed at iteration {len(history)}")
    gc = cfg.game()
    t0 = time.time()
    for step in range(len(history), cfg.steps):
        batch = _to(next(batches), device)
        b = batch["inputs"].shape[0]
        masks, labels = episode_masks(model.layout, b, gc, loader.dataset.lat, loader.dataset.lon, gen)
        obs_gen = torch.Generator().manual_seed(int(torch.randint(2 ** 31, (1,), generator=gen)))
        logs: Dict[str, float] = {"step": step}
        # ---- world model: the real (masked) rollouts with the current actor in the
        # loop, one sample at a time and stopped before a blow-up (play_memory)
        model.train(cfg.train_model)
        if opt_m is not None:
            opt_m.zero_grad(set_to_none=True)
        keep: list = []
        rec_f, rec_a, rec_r = [], [], []
        loss_sum, hid, alls, steps_done = 0.0, [], [], 0
        for i in range(b):
            one = {k: (v[i:i + 1] if torch.is_tensor(v) else v) for k, v in batch.items()}
            record: list = []
            with torch.set_grad_enabled(cfg.train_model):
                h, a = play_memory(model, ctrl, one, masks[i:i + 1], cfg.horizon, control=acting,
                                   explore=cfg.explore if acting else 0.0, obs_prob=cfg.obs_prob, gen=obs_gen,
                                   record=record, keep=keep, noise=noise, truncate=True,
                                   truncate_above=cfg.cap)
            done = h.shape[1]  # the states kept up to a blow-up are finite
            steps_done += done
            if done == 0:
                continue
            rewards = step_rewards(h, a, cfg.masked_weight)  # [1, K'] incl. the control cost
            loss = -rewards.sum() / (b * cfg.horizon)
            if cfg.train_model:
                loss.backward()
            loss_sum += float(loss.detach())
            hid.append(h.detach())
            alls.append(a.detach())
            if acting and cfg.imagination == "blind":
                rec_f += [f for f, _, _ in record]
                rec_a += [x for _, x, _ in record]
                rec_r += list((rewards.detach()[0] + torch.stack([c for _, _, c in record])[:, 0]
                               * ctrl.cost_weight).unbind())  # the error part: the cost is known
        logs.update(hidden=float(torch.cat(hid, 1).mean()) if hid else float("nan"),
                    all=float(torch.cat(alls, 1).mean()) if alls else float("nan"),
                    blowups=b * cfg.horizon - steps_done, loss_model=loss_sum)
        if acting and rec_f:
            target = symlog(torch.stack(rec_r))
            rloss = (heads.reward(torch.cat([torch.cat(rec_f), torch.cat(rec_a)], 1))[:, 0] - target).square().mean()
            rloss.backward()
            logs["reward_loss"] = float(rloss.detach())
        if opt_m is not None:
            params = [p for g in opt_m.param_groups for p in g["params"]]
            gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            logs["grad_model"] = float(gnorm)
            if torch.isfinite(gnorm):
                opt_m.step()
            else:
                logs["skipped"] = 1.0
        model.eval()
        # ---- actor and critic: imagined episodes in the current world model
        if acting and (keep or cfg.imagination == "snapshot"):
            if cfg.imagination == "snapshot":
                # from a masked snapshot (memory 0, as a test episode starts), with ERA5
                # re-observations and exact rewards; the episode ends at the horizon
                i = int(torch.randint(b, (1,), generator=gen))
                one = {k: (v[i:i + 1] if torch.is_tensor(v) else v) for k, v in batch.items()}
                record = []
                draws = torch.Generator().manual_seed(int(torch.randint(2 ** 31, (1,), generator=gen)))
                with torch.no_grad():
                    h_i, a_i = play_memory(model, ctrl, one, masks[i:i + 1], cfg.horizon, members=cfg.members,
                                           explore=cfg.explore, obs_prob=cfg.obs_prob, gen=draws, record=record,
                                           noise=noise, cap=cfg.cap)
                rews_i = step_rewards(h_i, a_i, cfg.masked_weight).t()  # [K, M], incl. the control cost
                feats_i = torch.stack([f for f, _, _ in record] + [torch.zeros_like(record[0][0])])
                acts_i = torch.stack([x for _, x, _ in record])
                # a blown-up member (scored the cap from then on) ends at its last finite
                # step: its later steps leave the losses and its chain of returns
                dead_i = (h_i >= cfg.cap).t()  # [K, M], cumulative
                cont = torch.cat([(~dead_i[1:]).float(), torch.zeros_like(rews_i[:1])])
                cont[-1] = 0.0  # terminal: the game's score ends at the horizon
                live_i = (~dead_i).float()
                logs["imagined_blowups"] = float(dead_i.any(0).sum())
            else:
                pick = int(torch.randint(len(keep), (1,), generator=gen))
                start = keep[pick]
                sample = sum(1 for e in keep[:pick + 1] if e[2] == 0) - 1  # which sample's rollout it came from
                feats_i, acts_i, rews_i, cont = imagine(model, ctrl, heads, start,
                                                        masks[sample:sample + 1].to(device), cfg.imagine_horizon,
                                                        cfg.members, cfg.explore, noise, cfg.blowup_penalty)
                live_i = None
                logs["imagined_blowups"] = float((1 - cont[-1]).sum())
            steps_i = rews_i.shape[0]
            with torch.no_grad():
                values = heads.value(feats_i)
                ret = lambda_returns(rews_i, values, cfg.gamma, cfg.lam, cont)  # [T, N]
                alive = torch.cat([torch.ones_like(cont[:1]), cont[:-1]])  # the step's state exists
                if live_i is not None:
                    alive = alive * live_i
                n_alive = alive.sum()
                if n_alive > 0:
                    lo, hi = torch.quantile(ret[alive > 0], torch.tensor([0.05, 0.95], device=ret.device))
                    s_now = float(hi - lo)
                    if math.isfinite(s_now):  # a non-finite scale would never recover
                        scale = s_now if scale is None else cfg.return_ema * scale + (1 - cfg.return_ema) * s_now
                if cfg.advantage == "group":
                    adv = group_advantage(ret, alive)
                else:
                    adv = (ret - values[:-1]) / max(1.0, scale if scale is not None else 1.0)
            ctrl.explore = cfg.explore
            logp = torch.stack([ctrl.log_prob(feats_i[t], acts_i[t]) for t in range(steps_i)])
            loss_actor = -(adv * logp * alive).sum() / n_alive.clamp(min=1)
            v_pred = heads.critic(feats_i[:-1])[..., 0]
            with torch.no_grad():
                v_ema = heads.critic_ema(feats_i[:-1])[..., 0]
            loss_critic = (((v_pred - symlog(ret)).square() + cfg.critic_reg * (v_pred - v_ema).square())
                           * alive).sum() / n_alive.clamp(min=1)
            opt_a.zero_grad(set_to_none=True)
            loss_agent = loss_actor + loss_critic
            logs["grad_agent"] = float("nan")
            if n_alive > 0 and torch.isfinite(loss_agent):
                loss_agent.backward()
                gnorm_a = torch.nn.utils.clip_grad_norm_(list(ctrl.parameters()) + list(heads.critic.parameters()),
                                                         cfg.grad_clip)
                logs["grad_agent"] = float(gnorm_a)
                if torch.isfinite(gnorm_a):  # one non-finite update makes the agent NaN for good
                    opt_a.step()
                    heads.update_ema(cfg.critic_ema)
                else:
                    logs["skipped_agent"] = 1.0
            else:
                logs["skipped_agent"] = 1.0
            logs.update(loss_actor=float(loss_actor.detach()), loss_critic=float(loss_critic.detach()),
                        return_scale=scale if scale is not None else float("nan"),
                        imagined_return=float(ret[0].mean()), action_rms=float(ctrl.net(feats_i[0]).detach()
                                                                              .square().mean().sqrt()))
        for s in scheds:
            s.step()
        logs["time_s"] = time.time() - t0
        history.append(logs)
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            terms = " ".join(f"{k}={v:.4g}" for k, v in logs.items() if k not in ("step", "time_s"))
            log(f"[dreamer] it {step:>4} {terms} ({logs['time_s']:.0f}s) {labels}")
        if save is not None and ((step + 1) % save_every == 0 or step == cfg.steps - 1):
            save({"model": model.state_dict(), "controller": ctrl.state_dict() if acting else None,
                  "heads": heads.state_dict() if acting else None,
                  "opt_model": opt_m.state_dict() if opt_m is not None else None,
                  "opt_agent": opt_a.state_dict() if acting else None,
                  "schedulers": [s.state_dict() for s in scheds], "generator": gen.get_state(),
                  "noise": noise.get_state(), "history": history, "return_scale": scale})
    if ctrl is not None:
        ctrl.explore = 0.0
    model.eval()
    return history


# ------------------------------------------------------------- evaluation
# variant -> (memory knock-outs, actor on, model ("own" / "base"), re-observation rule)
VARIANTS = {
    "full": ((), True, "own", "posterior"),
    "no_actor": ((), False, "own", "posterior"),
    "no_memory": (("memory",), True, "own", "posterior"),
    "no_innovation": (("innovation",), True, "own", "posterior"),
    "no_readout": (("readout",), True, "own", "posterior"),
    "no_spread": (("spread",), True, "own", "posterior"),
    "tau_fixed": (("adaptive",), True, "own", "posterior"),
    "nudging": ((), False, "base", 1.0),          # N1, classical nudging (the reference)
    "nudging_half": ((), False, "base", 0.5),
    "game_rule": ((), False, "base", "reset"),     # N1 under the stage 7/9 rule
}
REFERENCE = "nudging"


@torch.no_grad()
def evaluate_memory_game(model, ctrl: Optional[MemoryController], base, dataset, indices: Sequence[int],
                         kinds: Sequence[str], horizon: int, obs_prob: float, device="cpu",
                         variants: Sequence[str] = tuple(VARIANTS), seed: int = 0,
                         region_fraction: float = 0.25, log=None, cap: float = 10.0,
                         progress_path: Optional[str] = None) -> Dict[str, object]:
    """Masked-game errors per kind and lead for the trained model (`model`,
    `ctrl`), its knock-outs and the baselines on the stage 9 model (`base`),
    paired over initialisations; changes are relative to nudging on `base`.
    A blown-up episode (or an error above `cap`) scores `cap` from then on
    and is counted in "blowups"
    (`base` None: no baselines and no changes; the per-initialisation errors
    let a report pair runs with the same `indices` and `seed`). Also returns
    the learned timescales (log10 hours) of the full model. With `progress_path`
    every finished initialisation is saved there and an existing file is
    continued (the evaluations outlive the 2 h background limit)."""
    import os
    model.eval()
    if base is not None:
        base.eval()
    variants = [REFERENCE] + [v for v in variants if v != REFERENCE]
    out: Dict[str, object] = {"lead_h": [6 * (k + 1) for k in range(horizon)], "obs_prob": obs_prob,
                              "reference": REFERENCE, "variants": variants, "kinds": {}}
    tau_track: list = []
    saved: Dict[str, dict] = {}
    if progress_path is not None and os.path.exists(progress_path):
        saved = torch.load(progress_path, weights_only=False)
        tau_track = list(saved.pop("_tau_track", []))
        if log is not None:
            log("[memory game eval] resumed: " + ", ".join(f"{k} {len(v['labels'])}" for k, v in saved.items()))
    for kind in kinds:
        prev = saved.get(kind, {"rows": {}, "labels": [], "blowups": {}})
        rows: Dict[str, List[Tensor]] = {k: list(v) for k, v in prev["rows"].items()}
        labels = list(prev["labels"])
        blowups: Dict[str, int] = dict(prev["blowups"])
        for n, i in enumerate(indices):
            if n < len(labels):
                continue
            item = dataset[int(i)]
            batch = {k: v[None].to(device) for k, v in item.items()}
            gen = torch.Generator().manual_seed(seed + 1000 * n)
            if kind == "none":
                mask, label = torch.zeros(len(model.layout.dynamic), len(dataset.lat), len(dataset.lon)), "none"
            else:
                mask, label = sample_mask(model.layout, kind, dataset.lat, dataset.lon, gen, region_fraction)
            labels.append(label)
            for name in variants:
                knock, act, which, rule = VARIANTS[name]
                m = model if which == "own" else base
                if m is None:
                    continue
                if m.memory is not None:
                    m.memory.knock = set(knock)
                    m.memory.track = tau_track if (name == "full" and m is model) else None
                h, a = play_memory(m, ctrl if which == "own" else None, batch, mask[None], horizon,
                                   control=act, obs_prob=obs_prob, rule=rule,
                                   gen=torch.Generator().manual_seed(seed + 7919 * n), cap=cap)
                blowups[name] = blowups.get(name, 0) + int((h >= cap).any())
                if m.memory is not None:
                    m.memory.knock, m.memory.track = set(), None
                rows.setdefault(f"{name}_hidden", []).append(h[0])
                rows.setdefault(f"{name}_all", []).append(a[0])
            if progress_path is not None:
                saved[kind] = {"rows": {k: [x.cpu() for x in v] for k, v in rows.items()}, "labels": labels,
                               "blowups": blowups}
                torch.save({**saved, "_tau_track": tau_track}, progress_path + ".tmp")
                os.replace(progress_path + ".tmp", progress_path)
            if log is not None and (n + 1) % 10 == 0:
                log(f"[memory game eval] {kind}: {n + 1}/{len(indices)}")
        per = {k: torch.stack(v).cpu().numpy() for k, v in rows.items()}
        res = {"labels": labels, "blowups": blowups, "mse": {k: v.mean(0).tolist() for k, v in per.items()},
               "per_init": {k: v.tolist() for k, v in per.items()}, "change": {}}
        if f"{REFERENCE}_hidden" in per:
            for name in variants:
                if name == REFERENCE or f"{name}_hidden" not in per:
                    continue
                for part in ("hidden", "all"):
                    ref = per[f"{REFERENCE}_{part}"]
                    rel = (per[f"{name}_{part}"] - ref) / ref.mean(0)
                    idx = np.random.default_rng(seed).integers(0, len(rel), (2000, len(rel)))
                    means = rel[idx].mean(1)
                    res["change"][f"{name}_{part}"] = {"mean": rel.mean(0).tolist(),
                                                       "lo": np.percentile(means, 2.5, 0).tolist(),
                                                       "hi": np.percentile(means, 97.5, 0).tolist()}
        out["kinds"][kind] = res
    if tau_track:
        k = model.memory.channels
        z = torch.stack(tau_track)
        z = z[torch.isfinite(z).all(1)].mean(0)  # [H + 3 + 3K] averaged over the finite updates
        # (a blown-up episode's updates are NaN and would have hidden every timescale)
        out["log10_tau_hours_zonal"] = z[:-3 * (k + 1)].tolist()
        out["log10_tau_hours_p10_p50_p90"] = z[-3 * (k + 1):-3 * k].tolist()
        out["log10_tau_hours_channel_p10_p50_p90"] = z[-3 * k:].view(k, 3).tolist()
    return out
