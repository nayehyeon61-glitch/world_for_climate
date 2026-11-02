"""Stage 16 board 1: the N1 climate-bias game (README_KO section 16; design doc in the README).

RAIN (Nath et al. 2024) lets a policy set a climate model's tunable constants from the model's
own state, the physics unchanged, with the reward the distance to an observed climate. Here the
environment is N1 running free for `days` days from an ERA5 analysis; once a day the policy
reads N1's state (band means of its zonal-mean T, q, u on 13 levels, t2m and msl, the day of
year and its previous action) and sets the 10 knobs of models/knobs.py for the next 24 h.

Reward of day d: minus the cos-lat weighted squared distance between N1's and ERA5's
zonal means (T, u, q on every level, t2m, msl; normalised units) averaged over the last
`smooth_days` days, divided by the default model's mean error on that day (`ref`, from
`calibrate`, so the default scores about -1), minus a small action cost. The last day adds the
whole-episode mean distance (the episode's climate bias). A run whose all-variable error passes
`cap` (or turns non-finite) blows up: -`penalty` and the episode ends for that sample.

Agents: TD3 (RAIN's best at bias correction), PPO (its best at RCE), CEM for the best constant
setting (Kconst, the control RAIN lacks), and Kgrad, the same deterministic actor trained by
back-propagation through N1 (N1 is differentiable; climlab is not).
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from ..models import knobs as K

LEVEL_VARS = ("temperature", "u_component_of_wind", "specific_humidity")
SURFACE_VARS = ("2m_temperature", "mean_sea_level_pressure")
BANDS = (("south", -70.0, -20.0), ("tropics", -20.0, 20.0), ("north", 20.0, 70.0))
N_ACT = len(K.KNOBS)


@dataclasses.dataclass
class GameConfig:
    days: int = 15
    steps_per_day: int = 4
    smooth_days: int = 5
    penalty: float = 10.0
    cap: float = 10.0               # all-variable normalised MSE of a 6 h step that counts as a blow-up
    action_cost: float = 0.01
    terminal_weight: float = 1.0
    sparse_lag: int = 0             # > 0: rewards delivered summed every `sparse_lag` days (RAIN v2)
    state: bool = True              # False: the policy sees only the day of year (Knostate)
    fixed_groups: tuple = ()        # knob groups held at the default (a = 0)


# ---------------------------------------------------------------- scoring
class Scorer:
    """Zonal means of the reward channels, band features, and the weighted distances."""

    def __init__(self, model):
        keys = [c.key for c in model.layout.targets]
        dyn = [c.key for c in model.layout.dynamic]
        levels = [int(l) for l in model.layout.task.pressure_levels]
        self.keys = [f"{v}@{l}" for v in LEVEL_VARS for l in levels] + list(SURFACE_VARS)
        self.keys = [k for k in self.keys if k in keys]
        self.idx = torch.tensor([keys.index(k) for k in self.keys])
        self.dyn_idx = torch.tensor([dyn.index(k) if k in dyn else -1 for k in self.keys])
        lat = np.asarray(model.lat, np.float64)
        w = np.cos(np.deg2rad(lat))
        self.w = torch.tensor(w / w.sum(), dtype=torch.float32)                       # [H]
        bands = np.stack([(lat >= lo) & (lat <= hi) for _, lo, hi in BANDS]).astype(np.float64) * w
        self.bands = torch.tensor(bands / bands.sum(1, keepdims=True), dtype=torch.float32)  # [3, H]
        self.all_w = self.w                                                           # cap metric
        self.std = model.std.detach().cpu()[self.idx].float()
        self.n = len(self.keys)

    def zonal(self, pred: Tensor) -> Tensor:
        """[B, C, H, W] normalised targets -> [B, Cr, H]."""
        return pred[:, self.idx.to(pred.device)].float().mean(-1)

    def zonal_inputs(self, batch) -> Tensor:
        x = batch["inputs"][:, -1]
        out = torch.zeros(x.shape[0], self.n, x.shape[-2], device=x.device)
        for j, i in enumerate(self.dyn_idx.tolist()):
            if i >= 0:
                out[:, j] = x[:, i].float().mean(-1)
        return out

    def features(self, zm: Tensor) -> Tensor:
        """[B, Cr, H] -> band means [B, 3 Cr]."""
        return torch.einsum("bch,kh->bkc", zm, self.bands.to(zm.device)).flatten(1)

    def error(self, d: Tensor) -> Tensor:
        """Zonal-mean differences [B, Cr, H] -> weighted mean square [B]."""
        return (d.square() * self.w.to(d.device)).sum(-1).mean(-1)

    def cap_error(self, pred: Tensor, truth: Tensor) -> Tensor:
        """All-variable lat-weighted MSE of a 6 h step [B]."""
        w = self.all_w.to(pred.device)[None, None, :, None]
        return ((pred.float() - truth.float()).square() * w).sum(-2).mean((-1, -2))

    def bias_phys(self, d: Tensor) -> Dict[str, Tensor]:
        """Episode-mean zonal differences [B, Cr, H] -> RMS per variable in physical units [B]."""
        phys = d * self.std.to(d.device)[None, :, None]
        out = {}
        for v in LEVEL_VARS + SURFACE_VARS:
            j = [i for i, k in enumerate(self.keys) if k.split("@")[0] == v]
            if j:
                out[v] = torch.sqrt((phys[:, j].square() * self.w.to(d.device)).sum(-1).mean(-1))
        return out


def day_of_year(init_seconds: Tensor) -> Tensor:
    t = init_seconds.double() / 86400.0
    return (t % 365.2425).float()


def observation(scorer: Scorer, zm: Tensor, doy: Tensor, prev: Tensor, cfg: GameConfig,
                norm: Optional[dict] = None) -> Tensor:
    f = scorer.features(zm)
    if norm is not None:
        f = (f - norm["mean"].to(f.device)) / norm["std"].to(f.device)
    if not cfg.state:
        f = torch.zeros_like(f)
    ang = 2 * math.pi * doy.to(f.device) / 365.25
    return torch.cat([torch.nan_to_num(f, 0.0, 10.0, -10.0).clamp(-10, 10), torch.sin(ang)[:, None],
                      torch.cos(ang)[:, None], prev.to(f.device)], 1)


def obs_dim(scorer: Scorer) -> int:
    return 3 * scorer.n + 2 + N_ACT


# ---------------------------------------------------------------- the episode
@torch.no_grad()
def play(model, batch, policy: Callable[[Tensor, int], Tensor], cfg: GameConfig, scorer: Scorer,
         ref: Optional[dict] = None, norm: Optional[dict] = None) -> Dict[str, Tensor]:
    """One batch of episodes. policy(obs [B, D], day) -> action [B, 10] (raw; the knobs clamp it).
    Returns per day [days, B]: obs, act (raw), rew, valid (the sample was alive at the start of the
    day), done; next_obs; raw daily error `err`; blown_day [B] (-1 = never); bias (physical RMS of
    the episode-mean zonal difference per variable) and the daily mean action."""
    days, spd = cfg.days, cfg.steps_per_day
    b = batch["targets"].shape[0]
    mask = 1.0 - K.group_mask(cfg.fixed_groups) if cfg.fixed_groups else torch.ones(N_ACT)
    state, ctx = model.begin(batch)
    zm_last = scorer.zonal_inputs(batch)
    doy = day_of_year(batch["init_time"])
    prev = torch.zeros(b, N_ACT)
    alive = torch.ones(b, dtype=torch.bool)
    blown_day = torch.full((b,), -1)
    diffs, out = [], {k: [] for k in ("obs", "act", "rew", "valid", "done", "err")}
    for d in range(days):
        obs = observation(scorer, zm_last, doy + d, prev, cfg, norm)
        raw = policy(obs, d).detach().float().cpu()
        act = raw.clamp(-1, 1) * mask
        K.apply(model, act.to(batch["targets"].device))
        acc_p = acc_t = 0.0
        blew = torch.zeros(b, dtype=torch.bool)
        try:
            for j in range(spd):
                k = d * spd + j
                state, precip = model.advance(state, ctx, k)
                pred = model.predict(state, ctx, k, precip)
                truth = batch["targets"][:, k]
                finite = torch.isfinite(pred).flatten(1).all(1).cpu()
                p0 = torch.nan_to_num(pred.float(), 0.0, 0.0, 0.0)
                blew |= ~finite | (scorer.cap_error(p0, truth).cpu() > cfg.cap)
                acc_p = acc_p + scorer.zonal(p0).cpu()
                acc_t = acc_t + scorer.zonal(truth).cpu()
        finally:
            K.apply(model, None)
        zm_p, zm_t = acc_p / spd, acc_t / spd
        diffs.append(torch.where(alive[:, None, None], zm_p - zm_t, torch.zeros_like(zm_p)))
        lo = max(0, d - cfg.smooth_days + 1)
        e = scorer.error(torch.stack(diffs[lo:]).mean(0))
        r_ref = ref["day"][min(d, len(ref["day"]) - 1)] if ref else 1.0
        rew = -e / r_ref - cfg.action_cost * act.square().mean(1)
        if d == days - 1:
            t_ref = ref["term"] if ref else 1.0
            rew = rew - cfg.terminal_weight * scorer.error(torch.stack(diffs).mean(0)) / t_ref
        newly = blew & alive
        rew = torch.where(newly, torch.full_like(rew, -cfg.penalty), rew)
        rew = torch.where(alive, rew, torch.zeros_like(rew))
        out["obs"].append(obs.cpu())
        out["act"].append(raw)
        out["rew"].append(rew)
        out["valid"].append(alive.clone())
        out["done"].append(newly | torch.full_like(newly, d == days - 1))
        out["err"].append(torch.where(alive & ~newly, e, torch.full_like(e, float("nan"))))
        blown_day[newly] = d
        alive &= ~newly
        zm_last = torch.where(alive[:, None, None], zm_p, zm_last)
        prev = act
    next_last = observation(scorer, zm_last, doy + days, prev, cfg, norm).cpu()
    res = {k: torch.stack(v) for k, v in out.items()}
    res["next_obs"] = torch.cat([res["obs"][1:], next_last[None]])
    if cfg.sparse_lag > 0:
        res["rew"] = sparse_rewards(res["rew"], res["done"], cfg.sparse_lag)
    n_alive = torch.stack([v.float() for v in out["valid"]]).sum(0).clamp_min(1)[:, None, None]
    res["bias"] = scorer.bias_phys(torch.stack(diffs).sum(0) / n_alive)
    res["bias"]["all"] = scorer.error(torch.stack(diffs).sum(0) / n_alive)
    res["blown_day"] = blown_day
    return res


def sparse_rewards(rew: Tensor, done: Tensor, lag: int) -> Tensor:
    """RAIN v2: the rewards of each `lag`-day block are delivered at its end (or at the end of the episode)."""
    out = torch.zeros_like(rew)
    acc = torch.zeros_like(rew[0])
    ended = torch.zeros_like(done[0])
    for d in range(rew.shape[0]):
        acc = acc + torch.where(ended, torch.zeros_like(acc), rew[d])
        deliver = (((d + 1) % lag) == 0) | done[d]
        out[d] = torch.where(deliver & ~ended, acc, torch.zeros_like(acc))
        acc = torch.where(deliver, torch.zeros_like(acc), acc)
        ended = ended | done[d]
    return out


def returns(ep: Dict[str, Tensor]) -> Tensor:
    return (ep["rew"] * ep["valid"].float()).sum(0)


# ---------------------------------------------------------------- networks and agents
def mlp(d_in: int, d_out: int, hidden: int = 64, layers: int = 2) -> nn.Sequential:
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, d_out))
    return nn.Sequential(*mods)


class Actor(nn.Module):
    """Deterministic policy, tanh-bounded, starting near the default (a = 0)."""

    def __init__(self, d_obs: int, hidden: int = 64):
        super().__init__()
        self.net = mlp(d_obs, N_ACT, hidden)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: Tensor) -> Tensor:
        return torch.tanh(self.net(obs))


class TD3:
    def __init__(self, d_obs: int, hidden: int = 64, lr: float = 1e-3, gamma: float = 0.9, tau: float = 0.01,
                 policy_noise: float = 0.2, noise_clip: float = 0.5, explore: float = 0.3, delay: int = 2,
                 batch: int = 128, updates: int = 32, capacity: int = 100_000, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.actor = Actor(d_obs, hidden)
        self.critics = nn.ModuleList([mlp(d_obs + N_ACT, 1, hidden) for _ in range(2)])
        self.actor_t, self.critics_t = copy.deepcopy(self.actor), copy.deepcopy(self.critics)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critics.parameters(), lr=lr)
        self.hp = dict(gamma=gamma, tau=tau, policy_noise=policy_noise, noise_clip=noise_clip, explore=explore,
                       delay=delay, batch=batch, updates=updates)
        self.buf = {k: [] for k in ("o", "a", "r", "o2", "d")}
        self.capacity, self.n_updates, self.gen = capacity, 0, g

    def act(self, obs: Tensor, explore: bool) -> Tensor:
        with torch.no_grad():
            a = self.actor(obs)
        if explore:
            a = (a + self.hp["explore"] * torch.randn(a.shape, generator=self.gen)).clamp(-1, 1)
        return a

    def observe(self, ep: Dict[str, Tensor]) -> None:
        v = ep["valid"]
        for k, src in (("o", ep["obs"]), ("a", ep["act"].clamp(-1, 1)), ("r", ep["rew"]),
                       ("o2", ep["next_obs"]), ("d", ep["done"].float())):
            self.buf[k].append(src[v])
            self.buf[k] = [torch.cat(self.buf[k])[-self.capacity:]]

    def update(self) -> Dict[str, float]:
        o, a, r, o2, d = (self.buf[k][0] for k in ("o", "a", "r", "o2", "d"))
        hp, n = self.hp, len(o)
        if n < 16:
            return {}
        lc = la = 0.0
        for _ in range(hp["updates"]):
            i = torch.randint(0, n, (min(hp["batch"], n),), generator=self.gen)
            with torch.no_grad():
                noise = (hp["policy_noise"] * torch.randn(a[i].shape, generator=self.gen)).clamp(
                    -hp["noise_clip"], hp["noise_clip"])
                a2 = (self.actor_t(o2[i]) + noise).clamp(-1, 1)
                q2 = torch.min(*[c(torch.cat([o2[i], a2], 1)) for c in self.critics_t]).squeeze(1)
                y = r[i] + hp["gamma"] * (1 - d[i]) * q2
            loss_c = sum(((c(torch.cat([o[i], a[i]], 1)).squeeze(1) - y) ** 2).mean() for c in self.critics)
            self.opt_c.zero_grad()
            loss_c.backward()
            self.opt_c.step()
            lc = float(loss_c)
            self.n_updates += 1
            if self.n_updates % hp["delay"] == 0:
                loss_a = -self.critics[0](torch.cat([o[i], self.actor(o[i])], 1)).mean()
                self.opt_a.zero_grad()
                loss_a.backward()
                self.opt_a.step()
                la = float(loss_a)
                with torch.no_grad():
                    for net, tgt in ((self.actor, self.actor_t), (self.critics, self.critics_t)):
                        for p, pt in zip(net.parameters(), tgt.parameters()):
                            pt.mul_(1 - hp["tau"]).add_(hp["tau"] * p)
        return {"critic": lc, "actor": la, "buffer": n}

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critics": self.critics.state_dict(),
                "actor_t": self.actor_t.state_dict(), "critics_t": self.critics_t.state_dict(),
                "opt_a": self.opt_a.state_dict(), "opt_c": self.opt_c.state_dict(), "buf": self.buf,
                "n_updates": self.n_updates, "gen": self.gen.get_state()}

    def load_state_dict(self, sd: dict) -> None:
        for k in ("actor", "critics", "actor_t", "critics_t", "opt_a", "opt_c"):
            getattr(self, k).load_state_dict(sd[k])
        self.buf, self.n_updates = sd["buf"], sd["n_updates"]
        self.gen.set_state(sd["gen"])


class PPO:
    def __init__(self, d_obs: int, hidden: int = 64, lr: float = 3e-4, gamma: float = 0.9, lam: float = 0.95,
                 clip: float = 0.2, epochs: int = 10, minibatch: int = 32, log_std: float = math.log(0.4),
                 seed: int = 0):
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.actor = Actor(d_obs, hidden)
        self.log_std = nn.Parameter(torch.full((N_ACT,), log_std))
        self.critic = mlp(d_obs, 1, hidden)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + [self.log_std] + list(self.critic.parameters()),
                                    lr=lr)
        self.hp = dict(gamma=gamma, lam=lam, clip=clip, epochs=epochs, minibatch=minibatch)

    def act(self, obs: Tensor, explore: bool) -> Tensor:
        with torch.no_grad():
            mu = self.actor(obs)
            if not explore:
                return mu
            return mu + self.log_std.exp() * torch.randn(mu.shape, generator=self.gen)

    def _logp(self, obs: Tensor, act: Tensor) -> Tensor:
        mu = self.actor(obs)
        std = self.log_std.exp()
        return (-0.5 * ((act - mu) / std) ** 2 - self.log_std - 0.5 * math.log(2 * math.pi)).sum(-1)

    def update(self, ep: Dict[str, Tensor]) -> Dict[str, float]:
        hp = self.hp
        obs, act, rew, valid, done = ep["obs"], ep["act"], ep["rew"], ep["valid"].float(), ep["done"].float()
        days, b = rew.shape
        with torch.no_grad():
            v = self.critic(obs.flatten(0, 1)).view(days, b)
            v_last = self.critic(ep["next_obs"][-1]).squeeze(1)
            old = self._logp(obs.flatten(0, 1), act.flatten(0, 1)).view(days, b)
        adv = torch.zeros(days, b)
        gae = torch.zeros(b)
        for d in reversed(range(days)):
            nxt = v_last if d == days - 1 else v[d + 1]
            delta = rew[d] + hp["gamma"] * (1 - done[d]) * nxt - v[d]
            gae = delta + hp["gamma"] * hp["lam"] * (1 - done[d]) * gae
            adv[d] = gae * valid[d]
        ret = adv + v
        keep = valid.flatten().bool()
        o, a, lp0 = obs.flatten(0, 1)[keep], act.flatten(0, 1)[keep], old.flatten()[keep]
        A, R = adv.flatten()[keep], ret.flatten()[keep]
        A = (A - A.mean()) / (A.std() + 1e-8) if len(A) > 1 else A
        n = len(o)
        stats = {}
        for _ in range(hp["epochs"]):
            perm = torch.randperm(n, generator=self.gen)
            for s in range(0, n, hp["minibatch"]):
                i = perm[s:s + hp["minibatch"]]
                ratio = torch.exp(self._logp(o[i], a[i]) - lp0[i])
                pl = -torch.min(ratio * A[i], ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * A[i]).mean()
                vl = ((self.critic(o[i]).squeeze(1) - R[i]) ** 2).mean()
                loss = pl + 0.5 * vl
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 1.0)
                self.opt.step()
                stats = {"policy": float(pl.detach()), "value": float(vl.detach()),
                         "std": float(self.log_std.detach().exp().mean())}
        return stats

    def state_dict(self) -> dict:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "log_std": self.log_std.data,
                "opt": self.opt.state_dict(), "gen": self.gen.get_state()}

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.log_std.data.copy_(sd["log_std"])
        self.opt.load_state_dict(sd["opt"])
        self.gen.set_state(sd["gen"])


class CEM:
    """The best constant setting (Kconst): a diagonal Gaussian over a in [-1, 1]^10, refitted to the
    elite of each generation (one generation = one batch of episodes, one constant per sample)."""

    def __init__(self, sigma: float = 0.5, elite: float = 0.4, smooth: float = 0.7, floor: float = 0.05,
                 seed: int = 0):
        self.mean = torch.zeros(N_ACT)
        self.std = torch.full((N_ACT,), sigma)
        self.hp = dict(elite=elite, smooth=smooth, floor=floor)
        self.gen = torch.Generator().manual_seed(seed)
        self.best = (-float("inf"), torch.zeros(N_ACT))

    def ask(self, n: int) -> Tensor:
        a = self.mean + self.std * torch.randn(n, N_ACT, generator=self.gen)
        a[0] = self.mean                                    # the current mean is always scored
        return a.clamp(-1, 1)

    def tell(self, cand: Tensor, score: Tensor) -> Dict[str, float]:
        k = max(2, int(round(self.hp["elite"] * len(score))))
        top = torch.argsort(score, descending=True)[:k]
        s = self.hp["smooth"]
        self.mean = s * cand[top].mean(0) + (1 - s) * self.mean
        self.std = (s * cand[top].std(0) + (1 - s) * self.std).clamp_min(self.hp["floor"])
        if float(score[top[0]]) > self.best[0]:
            self.best = (float(score[top[0]]), cand[top[0]].clone())
        return {"elite_score": float(score[top].mean()), "std": float(self.std.mean())}

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "gen": self.gen.get_state(), "best": self.best}

    def load_state_dict(self, sd: dict) -> None:
        self.mean, self.std, self.best = sd["mean"], sd["std"], sd["best"]
        self.gen.set_state(sd["gen"])


def grad_loss(model, batch, actor: Actor, cfg: GameConfig, scorer: Scorer, ref: Optional[dict] = None,
              norm: Optional[dict] = None) -> Tensor:
    """Kgrad: the episode's (smoothed daily) distance as a differentiable loss of the actor's parameters,
    through N1 (whose own parameters stay frozen). The observation is detached."""
    days, spd = cfg.days, cfg.steps_per_day
    b = batch["targets"].shape[0]
    mask = (1.0 - K.group_mask(cfg.fixed_groups)) if cfg.fixed_groups else torch.ones(N_ACT)
    state, ctx = model.begin(batch)
    zm_last = scorer.zonal_inputs(batch).detach()
    doy = day_of_year(batch["init_time"])
    prev = torch.zeros(b, N_ACT)
    diffs, loss = [], 0.0
    for d in range(days):
        obs = observation(scorer, zm_last.cpu(), doy + d, prev, cfg, norm).detach()
        act = actor(obs) * mask
        K.apply(model, act.to(batch["targets"].device))
        acc_p = acc_t = 0.0
        try:
            for j in range(spd):
                k = d * spd + j
                state, precip = model.advance(state, ctx, k)
                pred = model.predict(state, ctx, k, precip)
                acc_p = acc_p + scorer.zonal(pred)
                acc_t = acc_t + scorer.zonal(batch["targets"][:, k])
        finally:
            K.apply(model, None)
        zm_p = acc_p / spd
        diffs.append(zm_p - acc_t / spd)
        lo = max(0, d - cfg.smooth_days + 1)
        r_ref = ref["day"][min(d, len(ref["day"]) - 1)] if ref else 1.0
        loss = loss + scorer.error(torch.stack(diffs[lo:]).mean(0)) / r_ref + \
            cfg.action_cost * act.square().mean(1).to(zm_p.device)
        zm_last = zm_p.detach()
        prev = act.detach()
    t_ref = ref["term"] if ref else 1.0
    loss = loss + cfg.terminal_weight * scorer.error(torch.stack(diffs).mean(0)) / t_ref
    return loss.mean()


# ---------------------------------------------------------------- data and loops
class Windows:
    """Batches of `days`-long windows from a GraphCastWindowDataset, 00 UTC initialisations."""

    def __init__(self, dataset, device="cpu"):
        self.ds, self.device = dataset, device

    def __len__(self) -> int:
        return len(self.ds)

    def batch(self, idx: Sequence[int]) -> Dict[str, Tensor]:
        items = [self.ds[int(i)] for i in idx]
        return {k: torch.stack([it[k] for it in items]).to(self.device) for k in items[0]}

    def spread(self, n: int) -> np.ndarray:
        return np.unique(np.linspace(0, len(self.ds) - 1, n).round().astype(int))


def calibrate(model, windows: Windows, cfg: GameConfig, scorer: Scorer, n: int = 16, batch: int = 8,
              log: Callable[[str], None] = print) -> dict:
    """K0 on `n` training windows: the default model's mean daily error (the reward scale), its
    episode-mean error, its blow-ups, and the mean / std of the observation features."""
    idx = windows.spread(n)
    errs, terms, feats, blown = [], [], [], []
    for s in range(0, len(idx), batch):
        b = windows.batch(idx[s:s + batch])
        zero = lambda o, d: torch.zeros(o.shape[0], N_ACT)
        ep = play(model, b, zero, cfg, scorer)
        errs.append(ep["err"])
        terms.append(ep["bias"]["all"])
        f = ep["obs"][..., :3 * scorer.n]
        feats.append(f[ep["valid"]])
        blown.append(ep["blown_day"])
        log(f"[calibrate] {s + len(b['targets'])}/{len(idx)} windows, blown {int((ep['blown_day'] >= 0).sum())}")
    err = torch.cat(errs, 1)
    f = torch.cat(feats)
    day = [float(np.nanmean(err[d].numpy())) for d in range(err.shape[0])]
    day = [max(x, 1e-6) if np.isfinite(x) else 1.0 for x in day]
    t = torch.cat(terms)
    return {"day": day, "term": max(float(np.nanmean(t.numpy())), 1e-6),
            "blown": int((torch.cat(blown) >= 0).sum()), "windows": int(len(idx)),
            "feat_mean": f.mean(0).tolist(), "feat_std": f.std(0).clamp_min(1e-3).tolist()}


def norm_from(calib: dict) -> dict:
    return {"mean": torch.tensor(calib["feat_mean"]), "std": torch.tensor(calib["feat_std"])}


def save_atomic(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train_agent(kind: str, model, windows: Windows, cfg: GameConfig, scorer: Scorer, calib: dict, out: str,
                iters: int, batch: int = 8, seed: int = 0, lr: Optional[float] = None,
                log: Callable[[str], None] = print) -> dict:
    """td3 | ppo | cem | grad. Resumable: out/progress.pt after every iteration; out/agent.pt at the end."""
    os.makedirs(out, exist_ok=True)
    ref, norm = {"day": calib["day"], "term": calib["term"]}, norm_from(calib)
    d_obs = obs_dim(scorer)
    if kind == "td3":
        agent = TD3(d_obs, seed=seed, **({"lr": lr} if lr else {}))
    elif kind == "ppo":
        agent = PPO(d_obs, seed=seed, **({"lr": lr} if lr else {}))
    elif kind == "cem":
        agent = CEM(seed=seed)
    elif kind == "grad":
        torch.manual_seed(seed)
        agent = Actor(d_obs)
        opt = torch.optim.Adam(agent.parameters(), lr=lr or 1e-3)
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    hist: List[dict] = []
    start = 0
    prog = os.path.join(out, "progress.pt")
    if os.path.exists(prog):
        sd = torch.load(prog, weights_only=False)
        agent.load_state_dict(sd["agent"])
        if kind == "grad":
            opt.load_state_dict(sd["opt"])
        hist, start = sd["hist"], sd["iter"]
        rng.bit_generator.state = sd["rng"]
        log(f"[{kind}] resuming at iteration {start}")
    for it in range(start, iters):
        if kind == "cem":   # every candidate on the same window: the ranking compares knobs, not weather
            idx = np.repeat(rng.choice(len(windows)), batch)
        else:
            idx = rng.choice(len(windows), size=min(batch, len(windows)), replace=False)
        b = windows.batch(idx)
        if kind == "grad":
            model.train(False)
            with torch.enable_grad():
                loss = grad_loss(model, b, agent, cfg, scorer, ref, norm)
                ok = bool(torch.isfinite(loss))
                if ok:
                    opt.zero_grad()
                    loss.backward()
                    gn = float(nn.utils.clip_grad_norm_(agent.parameters(), 1.0))
                    ok = math.isfinite(gn)
                    if ok:
                        opt.step()
            row = {"iter": it + 1, "loss": float(loss) if ok else float("nan"), "skipped": not ok}
        else:
            if kind == "td3":
                pol = lambda o, d: agent.act(o, explore=True)
            elif kind == "ppo":
                pol = lambda o, d: agent.act(o, explore=True)
            else:
                cand = agent.ask(len(idx))
                pol = lambda o, d: cand
            ep = play(model, b, pol, cfg, scorer, ref, norm)
            ret = returns(ep)
            if kind == "td3":
                agent.observe(ep)
                st = agent.update()
            elif kind == "ppo":
                st = agent.update(ep)
            else:
                st = agent.tell(cand, ret)
            row = {"iter": it + 1, "return": float(ret.mean()), "err": float(np.nanmean(ep["err"].numpy())),
                   "blown": int((ep["blown_day"] >= 0).sum()), "act_abs": float(ep["act"].clamp(-1, 1).abs().mean()),
                   **st}
        hist.append(row)
        log(f"[{kind}] " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
        sd = {"agent": agent.state_dict(), "hist": hist, "iter": it + 1, "rng": rng.bit_generator.state}
        if kind == "grad":
            sd["opt"] = opt.state_dict()
        save_atomic(sd, prog)
    final = {"kind": kind, "agent": agent.state_dict(), "d_obs": d_obs, "cfg": dataclasses.asdict(cfg)}
    save_atomic(final, os.path.join(out, "agent.pt"))
    with open(os.path.join(out, "train_history.json"), "w") as f:
        json.dump(hist, f, indent=1)
    return {"hist": hist}


def load_policy(kind: str, path: Optional[str], d_obs: int, seed: int = 0) -> Callable[[Tensor, int], Tensor]:
    """Deterministic evaluation policies: zero (K0), rand, const (CEM best mean), td3 / ppo / grad (actor)."""
    if kind == "zero":
        return lambda o, d: torch.zeros(o.shape[0], N_ACT)
    if kind == "rand":
        g = torch.Generator().manual_seed(seed)
        return lambda o, d: torch.rand(o.shape[0], N_ACT, generator=g) * 2 - 1
    sd = torch.load(path, weights_only=False)["agent"]
    if kind == "const":
        a = sd["mean"].clone()
        return lambda o, d: a.expand(o.shape[0], N_ACT)
    actor = Actor(d_obs)
    actor.load_state_dict(sd["actor"] if "actor" in sd else sd)
    actor.eval()
    return lambda o, d: actor(o).detach()


def evaluate(model, windows: Windows, policy, cfg: GameConfig, scorer: Scorer, calib: dict, out: str,
             n: int = 31, batch: int = 8, log: Callable[[str], None] = print) -> dict:
    """`n` initialisations spread over the windows, deterministic policy. Rows per initialisation;
    resumable per batch (out + '.progress')."""
    ref, norm = {"day": calib["day"], "term": calib["term"]}, norm_from(calib)
    idx = windows.spread(n)
    prog = out + ".progress"
    res = json.load(open(prog)) if os.path.exists(prog) else {"rows": []}
    done = {r["init"] for r in res["rows"]}
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        b = windows.batch(chunk)
        inits = [str(np.datetime64(int(t), "s")) for t in b["init_time"].cpu().numpy()]
        if all(i in done for i in inits):
            continue
        ep = play(model, b, policy, cfg, scorer, ref, norm)
        ret = returns(ep)
        for j, init in enumerate(inits):
            res["rows"].append({
                "init": init, "blown_day": int(ep["blown_day"][j]), "return": float(ret[j]),
                "err": [None if not np.isfinite(x) else float(x) for x in ep["err"][:, j].tolist()],
                "bias": {k: float(v[j]) for k, v in ep["bias"].items()},
                "act_mean": ep["act"][:, j].clamp(-1, 1).mean(0).tolist(),
                "act_std": ep["act"][:, j].clamp(-1, 1).std(0, correction=0).tolist()})
        with open(prog, "w") as f:
            json.dump(res, f)
        log(f"[eval] {len(res['rows'])}/{len(idx)}: mean return {np.mean([r['return'] for r in res['rows']]):.3f}, "
            f"blown {sum(r['blown_day'] >= 0 for r in res['rows'])}")
    rows = res["rows"]
    summary = {"inits": len(rows), "blown": sum(r["blown_day"] >= 0 for r in rows),
               "return": float(np.mean([r["return"] for r in rows])),
               "bias": {k: float(np.mean([r["bias"][k] for r in rows if r["blown_day"] < 0] or [np.nan]))
                        for k in rows[0]["bias"]},
               "knobs": dict(zip(K.KNOBS, np.mean([r["act_mean"] for r in rows], 0).round(4).tolist()))}
    out_obj = {"summary": summary, "rows": rows, "cfg": dataclasses.asdict(cfg)}
    with open(out, "w") as f:
        json.dump(out_obj, f, indent=1)
    return out_obj
