"""Stage 12: active observation. The world model's action is where to observe.

감자 (10:04 KST 11/01): add an action element so the memory model becomes a
world model in the full sense, with climate prediction as the purpose. The
atmosphere cannot be acted on, but the observing system can: an agent decides
where to look, the memory model assimilates what it sees (the stage 10
posterior) and forecasts the rest. The reward is the forecast error, so the
agent learns where an observation is worth most to this model, with this
memory.

One episode = one initialisation and `horizon` 6 h steps. The start is the
fully observed snapshot (as a GraphCast initialisation; the memory is empty).
After each of the first `obs_steps` forecasts the rule picks `n` of the P
equal-area patches (4 latitude bands of equal area x 6 longitude sectors; a
budget of 25% = 6 patches); the chosen patches are re-observed with every
variable, the rest keep the forecast plus the memory's readout (the stage 10
posterior; on the stage 9 model without memory this is nudging with strength
1). The remaining steps run free. The score of every step is the
GraphCast-weighted all-point error of the forecast before the re-observation,
so the choice at step k is rewarded by the errors from step k + 1 on.

Rules:
    policy    the learned policy: per patch, a shared MLP reads the forecast's
              6 h change per variable group, the memory's readout, state and
              timescale in the patch, the patch's age (steps since it was last
              observed), its latitude and the step, with the global mean of
              the same features as context; a Plackett-Luce sample of n
              patches (Gumbel-top-k), the argmax n at test time
    random    n patches at random (the control: the policy starts here)
    age       the n patches observed longest ago (round-robin)
    readout   the n patches where the memory's readout |R(m)| is largest (the
              model's own estimate of where it is wrong; memory model only)
    tendency  the n patches where the forecast changed most in 6 h
    full      everything observed at every step (the upper bound)
    none      nothing observed (the cold free run; the lower bound)

Training (the frozen world model as in stage 7): `members` episodes from one
start with sampled choices, REINFORCE on the group-normalised reward-to-go of
each choice (train/prediction_game.reward_to_go_advantages, stages 7, 9, 11).
The last layer starts at zero, so the untrained policy is the random rule.

Tests: the game on 2020 (31 initialisations, 120 h: 48 h of observations
then 72 h free, the GraphCast leads) with a budget sweep, and the climate
frame (10 days of budgeted observations, then 4 weeks free, weekly means
against ERA5 and the climatology, as eval/memory_climate.py).
"""

from __future__ import annotations

import dataclasses
import math
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..eval.graphcast_metrics import HEADLINE_KEYS, acc_per_init, normalized_latitude_weights, weighted_mean
from .physics_game import _detach, _finite, _select, attach_loss
from .prediction_game import _repeat, errors, reward_to_go_advantages, variable_groups
from .trainer import _forever, _to

RULES = ("policy", "random", "age", "readout", "tendency", "full", "none")
FIXED = ("random", "age", "readout", "tendency")


# ------------------------------------------------------------------ patches
class Patches:
    """Equal-area patches of the sphere: `bands` latitude bands of equal area
    (edges at equal steps of sin lat) x `sectors` longitude sectors. Maps are
    built for the data grid (masks) and, when given, the model's Gaussian grid
    (the memory's fields)."""

    def __init__(self, lat, lon, lat_g=None, lon_g=None, bands: int = 4, sectors: int = 6):
        self.bands, self.sectors, self.n = bands, sectors, bands * sectors
        self.edges = np.degrees(np.arcsin(np.linspace(-1.0, 1.0, bands + 1)))
        self.id_data = self._ids(lat, lon)
        self.pool_data = self._pool(lat, self.id_data)
        self.id_gauss = self._ids(lat_g, lon_g) if lat_g is not None else None
        self.pool_gauss = self._pool(lat_g, self.id_gauss) if lat_g is not None else None
        mids = np.degrees(np.arcsin(0.5 * (np.sin(np.deg2rad(self.edges[:-1])) + np.sin(np.deg2rad(self.edges[1:])))))
        self.lat_centre = np.repeat(mids, sectors)
        self.lon_centre = np.tile((np.arange(sectors) + 0.5) * 360.0 / sectors, bands)
        self.sinlat = torch.tensor(np.sin(np.deg2rad(self.lat_centre)), dtype=torch.float32)

    def _ids(self, lat, lon) -> Tensor:
        lat, lon = np.asarray(lat, np.float64), np.asarray(lon, np.float64)
        band = np.clip(np.searchsorted(self.edges, lat, side="right") - 1, 0, self.bands - 1)
        sector = np.minimum((np.mod(lon, 360.0) // (360.0 / self.sectors)).astype(int), self.sectors - 1)
        return torch.tensor(band[:, None] * self.sectors + sector[None, :], dtype=torch.long)

    def _pool(self, lat, ids: Tensor) -> Tensor:
        """[P, H * W] area weights, normalised within each patch (a patch mean)."""
        w = np.repeat(np.cos(np.deg2rad(np.asarray(lat, np.float64)))[:, None], ids.shape[1], 1).flatten()
        onehot = nn.functional.one_hot(ids.flatten(), self.n).t().double() * torch.from_numpy(w)[None]
        return (onehot / onehot.sum(1, keepdim=True).clamp_min(1e-12)).float()

    def mean(self, x: Tensor, grid: str = "data") -> Tensor:
        """Patch means [..., P] of grid fields [..., H, W]."""
        pool = self.pool_data if grid == "data" else self.pool_gauss
        return x.flatten(-2) @ pool.to(x.device).t()

    def hidden_mask(self, choice: Tensor, channels: int) -> Tensor:
        """Chosen patches [N, n] -> game mask [N, channels, H, W] (1 = hidden = not chosen)."""
        chosen = torch.zeros(choice.shape[0], self.n, device=choice.device).scatter_(1, choice, 1.0)
        seen = chosen[:, self.id_data.flatten().to(choice.device)].view(-1, *self.id_data.shape)
        return (1 - seen)[:, None].expand(-1, channels, -1, -1).contiguous()

    def chosen(self, choice: Tensor) -> Tensor:
        """[N, P] indicator of the chosen patches."""
        return torch.zeros(choice.shape[0], self.n, device=choice.device).scatter_(1, choice, 1.0)


# ------------------------------------------------------------------- policy
@dataclasses.dataclass
class ObsPolicyConfig:
    bands: int = 4
    sectors: int = 6
    budget: float = 0.25          # fraction of the patches observed at each observation time
    hidden: int = 64
    use_state: bool = True        # the forecast's 6 h change per variable group
    use_memory: bool = True       # the memory's readout, state and timescale per patch (zero without memory)
    max_horizon: int = 8
    temperature: float = 1.0
    seed: int = 0


def sample_subset(logits: Tensor, n: int, gen: Optional[torch.Generator] = None,
                  deterministic: bool = False) -> Tensor:
    """n patches without replacement from the Plackett-Luce distribution of
    `logits` [N, P] (Gumbel-top-k), in sampling order [N, n]; the n largest
    logits when deterministic."""
    if deterministic:
        return logits.topk(n, -1).indices
    u = torch.rand(logits.shape, generator=gen).clamp(1e-20, 1 - 1e-7).to(logits.device)
    return (logits - torch.log(-torch.log(u))).topk(n, -1).indices


def subset_log_prob(logits: Tensor, choice: Tensor) -> Tensor:
    """log p of an ordered sample `choice` [N, n] under Plackett-Luce [N]."""
    lp = logits.new_zeros(logits.shape[0])
    avail = torch.ones_like(logits, dtype=torch.bool)
    for j in range(choice.shape[1]):
        c = choice[:, j:j + 1]
        masked = logits.masked_fill(~avail, float("-inf"))
        lp = lp + logits.gather(1, c)[:, 0] - torch.logsumexp(masked, 1)
        avail = avail.scatter(1, c, False)
    return lp


def choice_entropy(logits: Tensor) -> Tensor:
    """Entropy (nat) of the first pick's categorical distribution, per row."""
    p = torch.softmax(logits, -1)
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1)


class ObservationPolicy(nn.Module):
    """Per-patch logits from patch features (module docstring: rule `policy`)."""

    def __init__(self, layout, patches: Patches, cfg: ObsPolicyConfig):
        super().__init__()
        self.cfg, self.patches = cfg, patches
        groups = variable_groups(layout)
        self.group_names = sorted(groups)
        self.group_index = [groups[g] for g in self.group_names]
        self.n_local = len(self.group_names) + 3 + 1   # state groups, memory (readout, state, tau), age
        in_dim = 2 * self.n_local + 1 + cfg.max_horizon  # local + global context, sin lat, step
        torch.manual_seed(cfg.seed)
        self.net = nn.Sequential(nn.Linear(in_dim, cfg.hidden), nn.SiLU(), nn.Linear(cfg.hidden, cfg.hidden),
                                 nn.SiLU(), nn.Linear(cfg.hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)   # the untrained policy is the random rule
        self.random_actions = False

    def n_obs(self, budget: Optional[float] = None) -> int:
        b = self.cfg.budget if budget is None else budget
        return max(1, min(self.patches.n, int(round(b * self.patches.n))))

    def features(self, model, state: Dict[str, Tensor], ctx, k: int, forecast: Tensor, previous: Tensor,
                 age: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Patch features [N, P, F] at lead k + 1 and the raw maps the fixed rules
        rank (`tendency`, `readout`, `age`, `tau`; [N, P] each). `forecast` and
        `previous` are normalised dynamic fields [N, C_dyn, H, W] on the data grid."""
        pt = self.patches
        n = forecast.shape[0]
        d2 = (forecast - previous).float().square()
        groups = torch.stack([pt.mean(d2[:, idx].mean(1)) for idx in self.group_index], -1).sqrt()  # [N, P, G]
        raw = {"tendency": pt.mean(d2.mean(1)).sqrt(), "age": age}
        zeros = forecast.new_zeros(n, pt.n)
        readout, mem, tau = zeros, zeros, zeros
        if model.memory is not None and "mem" in state and pt.pool_gauss is not None:
            m = state["mem"].float()
            r = model.memory.readout(m)
            if r is not None:
                readout = pt.mean(r.float().square().mean(1), "gauss").sqrt()
            mem = pt.mean(m.square().mean(1), "gauss").sqrt()
            x, _, _ = model.column_inputs(state, ctx["static_g"], model.forcing_grid_at(ctx, k + 1))
            lt = torch.log10(model.memory.tau(x).float() / 3600.0).mean(-1)  # [N, H_g, W_g] log10 hours
            tau = pt.mean(lt, "gauss")
        raw.update(readout=readout, tau=tau)
        if not self.cfg.use_state:
            groups = torch.zeros_like(groups)
        memory = torch.stack([readout, mem, (tau - 1.5) / 1.5], -1) if self.cfg.use_memory \
            else forecast.new_zeros(n, pt.n, 3)
        local = torch.cat([groups, memory, (age / self.cfg.max_horizon)[..., None]], -1)  # [N, P, n_local]
        context = local.mean(1, keepdim=True).expand(-1, pt.n, -1)
        step = forecast.new_zeros(n, pt.n, self.cfg.max_horizon)
        step[..., min(k, self.cfg.max_horizon - 1)] = 1.0
        sinlat = pt.sinlat.to(forecast.device)[None, :, None].expand(n, -1, -1)
        return torch.cat([local, context, sinlat, step], -1), raw

    def logits(self, feats: Tensor) -> Tensor:
        return self.net(feats)[..., 0] / self.cfg.temperature


def choose(rule: str, raw: Dict[str, Tensor], logits: Optional[Tensor], n: int,
           gen: Optional[torch.Generator] = None, deterministic: bool = False) -> Tensor:
    """The rule's patches [N, n] (module docstring)."""
    if rule == "policy":
        return sample_subset(logits, n, gen, deterministic)
    key = next(iter(raw.values()))
    noise = torch.rand(key.shape, generator=gen).to(key.device)  # random rule, and tie-breaking
    if rule == "random":
        return noise.topk(n, -1).indices
    if rule not in ("age", "readout", "tendency"):
        raise ValueError(f"unknown rule {rule}")
    score = raw[rule].float()
    scale = score.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    return (score / scale + 1e-3 * noise).topk(n, -1).indices


# ------------------------------------------------------------------ episode
def play_observation(model, policy: ObservationPolicy, batch, rule: str, horizon: int, obs_steps: int,
                     members: int = 1, budget: Optional[float] = None, gen: Optional[torch.Generator] = None,
                     cap: float = 10.0, deterministic: bool = False, record: Optional[list] = None,
                     on_step: Optional[Callable[[int, Tensor, Tensor], None]] = None
                     ) -> Tuple[Tensor, List[Tensor], Dict[str, Tensor]]:
    """One episode per initial state, `members` copies each (module docstring).
    Returns the all-point errors [N, K] (a blown-up member scores `cap` from
    then on and restarts from the initial state), the choices per observation
    step ([N, n] each; empty for full / none) and info: `dead` [N], and for a
    memory model the patch means over the observation steps of the memory's
    log10 timescale and readout ("tau", "readout", [N, P]). `record` receives
    (features, choice, alive) per observation step for the policy gradient.
    `on_step(k, pred, dead)` sees every forecast."""
    attach_loss(model)
    pt = policy.patches
    n_obs = policy.n_obs(budget)
    rep = _repeat(batch, members)
    rep["init_time"] = batch["init_time"].repeat_interleave(members)
    t2d = model.layout.target_to_dynamic
    n = rep["inputs"].shape[0]
    zero = torch.zeros_like(rep["inputs"][:, -1])
    zero_t = zero.new_zeros(n, len(model.layout.targets), *zero.shape[-2:])
    if model.memory is not None:
        model.memory.control_tau = None
    with torch.no_grad():
        state, ctx = model.begin(rep)
        snap = rep["inputs"][:, -1]
        if model.memory is not None and rule != "none":
            state = model.posterior(state, ctx, snap, snap, zero)  # the memory takes in the full snapshot
        first, first_resid = _detach(state), ctx["resid"].detach()
        dead = torch.zeros(n, dtype=torch.bool, device=snap.device)
        age = snap.new_zeros(n, pt.n)
        previous = snap
        errs, choices = [], []
        tau_sum, read_sum, n_sum = None, None, 0
        for k in range(horizon):
            state, precip = model.advance(state, ctx, k)
            pred = model.predict(state, ctx, k, precip)
            _, e = errors(model, pred, rep["targets"][:, k], zero_t)
            bad = ~(torch.isfinite(e) & _finite(state))
            dead = dead | bad
            e = torch.where(dead, torch.full_like(e, cap), torch.nan_to_num(e, nan=cap).clamp(max=cap))
            if bad.any():  # restart blown-up members so the batch stays finite
                state = {key: _select(bad, first[key], v) for key, v in state.items()}
                ctx["resid"] = _select(bad, first_resid, ctx["resid"])
                pred = torch.nan_to_num(pred)
            errs.append(e)
            if on_step is not None:
                on_step(k, pred, dead)
            fc = pred[:, t2d]
            if k < obs_steps and k < horizon - 1 and rule != "none":
                truth = rep["targets"][:, k][:, t2d]
                raw = None
                if rule != "full" or model.memory is not None:  # full: features only for the memory's maps
                    feats, raw = policy.features(model, state, ctx, k, fc, previous, age)
                if rule == "full":
                    mask = zero
                else:
                    logits = policy.logits(feats) if rule == "policy" else None
                    if rule == "policy" and policy.random_actions:
                        choice = choose("random", raw, None, n_obs, gen)
                    else:
                        choice = choose(rule, raw, logits, n_obs, gen, deterministic)
                    mask = pt.hidden_mask(choice, fc.shape[1])
                    if record is not None:
                        record.append((feats, choice, ~dead))
                    choices.append(choice)
                    age = (age + 1) * (1 - pt.chosen(choice))
                if raw is not None and model.memory is not None:
                    tau_sum = raw["tau"] if tau_sum is None else tau_sum + raw["tau"]
                    read_sum = raw["readout"] if read_sum is None else read_sum + raw["readout"]
                    n_sum += 1
                state = model.posterior(state, ctx, fc, truth, mask)
                previous = (1 - mask) * truth + mask * fc
            else:
                previous = fc
    info = {"dead": dead}
    if n_sum:
        info.update(tau=tau_sum / n_sum, readout=read_sum / n_sum)
    return torch.stack(errs, 1), choices, info


# ----------------------------------------------------------------- training
@dataclasses.dataclass
class ActiveConfig:
    steps: int = 100
    members: int = 8
    horizon: int = 10        # 60 h
    obs_steps: int = 6       # choices after the first 6 forecasts (36 h), then a free run
    budget: float = 0.25
    lr: float = 1e-3
    warmup_steps: int = 10
    grad_clip: float = 1.0
    cap: float = 10.0
    entropy: float = 0.0     # bonus on the first pick's entropy
    log_every: int = 5
    seed: int = 0


def _schedule(cfg: ActiveConfig):
    def f(step):
        if cfg.warmup_steps and step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * step / max(cfg.steps, 1)))
    return f


def train_policy(model, policy: ObservationPolicy, loader: DataLoader, cfg: ActiveConfig, device="cpu",
                 log=print, progress: Optional[dict] = None, save: Optional[Callable[[dict], None]] = None,
                 save_every: int = 5) -> List[Dict[str, float]]:
    """REINFORCE on the frozen world model (module docstring); one start per
    iteration, `cfg.members` sampled episodes. `save` receives a resumable
    progress dict every `save_every` iterations."""
    attach_loss(model)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _schedule(cfg))
    gen = torch.Generator().manual_seed(cfg.seed)
    history: List[Dict[str, float]] = []
    batches = _forever(loader)
    if progress is not None:
        policy.load_state_dict(progress["policy"])
        opt.load_state_dict(progress["opt"])
        sched.load_state_dict(progress["sched"])
        gen.set_state(progress["generator"])
        history = list(progress["history"])
        for _ in range(len(history)):
            next(batches)
        log(f"[active] resumed at iteration {len(history)}")
    t0 = time.time()
    counts = torch.zeros(policy.patches.n)
    for step in range(len(history), cfg.steps):
        batch = _to(next(batches), device)
        one = {k: (v[:1] if torch.is_tensor(v) else v) for k, v in batch.items()}
        draws = torch.Generator().manual_seed(int(torch.randint(2 ** 31, (1,), generator=gen)))
        record: list = []
        policy.eval()
        e, choices, info = play_observation(model, policy, one, "policy", cfg.horizon, cfg.obs_steps,
                                            members=cfg.members, budget=cfg.budget, gen=draws, cap=cfg.cap,
                                            record=record)
        logs: Dict[str, float] = {"step": step, "error_obs": float(e[:, :cfg.obs_steps].mean()),
                                  "error_free": float(e[:, cfg.obs_steps:].mean()), "error": float(e.mean()),
                                  "blowups": int(info["dead"].sum())}
        policy.train()
        if record:
            adv = reward_to_go_advantages(-e[:, 1:], cfg.members)        # [M, K - 1]: rewards from k + 1 on
            feats = torch.stack([f for f, _, _ in record])              # [T, M, P, F]
            chs = torch.stack([c for _, c, _ in record])                # [T, M, n]
            alive = torch.stack([a for _, _, a in record]).float()      # [T, M]
            logits = policy.logits(feats)                               # [T, M, P]
            logp = torch.stack([subset_log_prob(logits[t], chs[t]) for t in range(len(record))])
            weight = adv[:, :len(record)].t() * alive
            loss = -(weight * logp).sum() / alive.sum().clamp(min=1)
            ent = choice_entropy(logits).mean()
            loss = loss - cfg.entropy * ent
            opt.zero_grad(set_to_none=True)
            logs["grad"] = float("nan")
            if alive.sum() > 0 and torch.isfinite(loss):
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
                logs["grad"] = float(gnorm)
                if torch.isfinite(gnorm):
                    opt.step()
                else:
                    logs["skipped"] = 1.0
            else:
                logs["skipped"] = 1.0
            with torch.no_grad():
                for c in chs:
                    counts += policy.patches.chosen(c).sum(0).cpu()
            logs.update(loss=float(loss.detach()), entropy=float(ent.detach()),
                        return_spread=float(adv[:, 0].std()) if cfg.members > 1 else 0.0)
        sched.step()
        logs["time_s"] = time.time() - t0
        history.append(logs)
        if cfg.log_every and (step % cfg.log_every == 0 or step == cfg.steps - 1):
            terms = " ".join(f"{k}={v:.4g}" for k, v in logs.items() if k not in ("step", "time_s"))
            log(f"[active] it {step:>4} {terms} ({logs['time_s']:.0f}s)")
        if save is not None and ((step + 1) % save_every == 0 or step == cfg.steps - 1):
            save({"policy": policy.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                  "generator": gen.get_state(), "history": history, "counts": counts.tolist()})
    policy.eval()
    return history


# --------------------------------------------------------------- evaluation
def _bootstrap(rel: np.ndarray, seed: int) -> Dict[str, object]:
    idx = np.random.default_rng(seed).integers(0, len(rel), (2000, len(rel)))
    means = rel[idx].mean(1)
    return {"mean": rel.mean(0).tolist(), "lo": np.percentile(means, 2.5, 0).tolist(),
            "hi": np.percentile(means, 97.5, 0).tolist()}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _headline(model, dataset):
    targets = list(model.layout.targets)
    keys = [k for k in HEADLINE_KEYS if k in [c.key for c in targets]]
    kidx = [[c.key for c in targets].index(k) for k in keys]
    lat_w = torch.tensor(normalized_latitude_weights(dataset.lat), dtype=torch.float32)
    return targets, keys, kidx, lat_w


def run_names(rules: Sequence[str], budgets: Sequence[float], base: bool, base_rules: Sequence[str]) -> List[Tuple[str, str, float, str]]:
    """(name, model, budget, rule) of the runs of the game evaluation."""
    out = []
    for which in (("memory", "n1") if base else ("memory",)):
        for budget in budgets:
            for rule in (rules if which == "memory" else base_rules):
                if rule in ("full", "none") and budget != budgets[0]:
                    continue
                if rule == "readout" and which == "n1":
                    continue
                name = f"{which}:{rule}" if budget == budgets[0] or rule in ("full", "none") \
                    else f"{which}:{rule}@{budget:g}"
                out.append((name, which, budget, rule))
    return out


def evaluate_active_obs(model, policy: ObservationPolicy, base, dataset, indices: Sequence[int], stats,
                        rules: Sequence[str] = RULES, budgets: Sequence[float] = (0.25,), horizon: int = 20,
                        obs_steps: int = 8, base_rules: Sequence[str] = ("policy", "random", "full", "none"),
                        device="cpu", seed: int = 0, cap: float = 10.0, log=None,
                        progress_path: Optional[str] = None, reference: str = "memory:random") -> Dict[str, object]:
    """The observation game on every initialisation: the memory model under every
    rule and budget, the stage 9 model `base` (no memory: nudging) under
    `base_rules`; paired draws per initialisation. Returns per-run errors per lead
    (GraphCast-weighted all-point MSE, and the headline variables' RMSE in
    physical units), blow-ups, selection frequencies per patch, the memory's
    timescale map per patch, changes against `reference` (mean over the free
    steps and over all steps, bootstrap over initialisations) and the rank
    correlation of each rule's frequencies with the timescale map."""
    model.eval()
    if base is not None:
        base.eval()
    targets, keys, kidx, lat_w = _headline(model, dataset)
    runs = run_names(rules, budgets, base is not None, base_rules)
    done: Dict[str, List[dict]] = {}
    if progress_path is not None and os.path.exists(progress_path):
        done = torch.load(progress_path, weights_only=False)
        if log is not None:
            log(f"[active eval] resumed: {len(next(iter(done.values()), []))} initialisations")
    recs = done.setdefault("recs", [])
    for n, i in enumerate(indices):
        if n < len(recs):
            continue
        item = dataset[int(i)]
        batch = {k: v[None].to(device) for k, v in item.items()}
        truth_p = stats.unnormalize(batch["targets"][0].float().cpu(), targets)  # [K, C, H, W]
        rec: Dict[str, object] = {"init": int(item["init_time"])}
        for name, which, budget, rule in runs:
            m = model if which == "memory" else base
            per = []

            def on_step(k, pred, dead, per=per):
                phys = stats.unnormalize(pred[0].float().cpu(), targets)
                per.append(weighted_mean((phys - truth_p[k]).square(), lat_w)[kidx].sqrt().tolist())

            e, choices, info = play_observation(m, policy, batch, rule, horizon, obs_steps, 1, budget,
                                                torch.Generator().manual_seed(seed + 7919 * n), cap,
                                                deterministic=(rule == "policy"), on_step=on_step)
            rec[name] = {"mse": e[0].cpu().tolist(), "rmse": per, "blown": bool(info["dead"][0]),
                         "choices": [c[0].cpu().tolist() for c in choices]}
            if "tau" in info:
                rec[name]["tau"] = info["tau"][0].cpu().tolist()
                rec[name]["readout_map"] = info["readout"][0].cpu().tolist()
        recs.append(rec)
        if progress_path is not None:
            torch.save(done, progress_path + ".tmp")
            os.replace(progress_path + ".tmp", progress_path)
        if log is not None:
            log(f"[active eval] {n + 1}/{len(indices)}")
    pt = policy.patches
    out: Dict[str, object] = {"lead_h": [6 * (k + 1) for k in range(horizon)], "obs_steps": obs_steps,
                              "horizon": horizon, "budgets": list(budgets), "keys": keys, "reference": reference,
                              "patches": {"bands": pt.bands, "sectors": pt.sectors, "lat_centre": pt.lat_centre.tolist(),
                                          "lon_centre": pt.lon_centre.tolist()},
                              "inits": [r["init"] for r in recs], "runs": {}, "change": {}, "spearman_tau": {}}
    mse = {name: np.array([r[name]["mse"] for r in recs]) for name, *_ in runs}
    for name, which, budget, rule in runs:
        rows = [r[name] for r in recs]
        freq = np.zeros(pt.n)
        steps = 0
        for r in rows:
            for c in r["choices"]:
                freq[c] += 1
                steps += 1
        freq = freq / max(steps, 1)
        p = freq / max(freq.sum(), 1e-12)
        out["runs"][name] = {"model": which, "rule": rule, "budget": budget, "mse": mse[name].mean(0).tolist(),
                             "rmse": {k: np.array([r["rmse"] for r in rows])[:, :, j].mean(0).tolist()
                                      for j, k in enumerate(keys)},
                             "blowups": int(sum(r["blown"] for r in rows)), "freq": freq.tolist(),
                             "freq_entropy": float(-(p[p > 0] * np.log(p[p > 0])).sum()) if steps else float("nan")}
    tau_rows = [r[name]["tau"] for name, which, _, rule in runs if which == "memory" and rule == "full"
                for r in recs if "tau" in r[name]]
    if tau_rows:
        out["tau_patch"] = np.mean(tau_rows, 0).tolist()
        out["readout_patch"] = np.mean([r[name]["readout_map"] for name, which, _, rule in runs
                                        if which == "memory" and rule == "full" for r in recs], 0).tolist()
        for name in out["runs"]:
            freq = np.array(out["runs"][name]["freq"])
            out["spearman_tau"][name] = _spearman(freq, np.array(out["tau_patch"])) if freq.sum() > 0 else float("nan")
    if reference in mse:
        ref = mse[reference]
        for name in mse:
            if name == reference:
                continue
            rel_free = (mse[name][:, obs_steps:].mean(1) - ref[:, obs_steps:].mean(1)) / ref[:, obs_steps:].mean()
            rel_all = (mse[name].mean(1) - ref.mean(1)) / ref.mean()
            rel_lead = (mse[name] - ref) / ref.mean(0)
            out["change"][name] = {"free": _bootstrap(rel_free[:, None], seed), "all": _bootstrap(rel_all[:, None], seed),
                                   "per_lead": rel_lead.mean(0).tolist()}
    return out


def evaluate_active_climate(model, policy: ObservationPolicy, base, dataset, indices: Sequence[int], climatology,
                            stats, rules: Sequence[str] = ("policy", "random", "full", "none"), budget: float = 0.25,
                            spinup: int = 40, weeks: int = 4, week: int = 28,
                            base_rules: Sequence[str] = ("policy", "random"), device="cpu", seed: int = 0,
                            cap: float = 10.0, log=None, progress_path: Optional[str] = None,
                            reference: str = "memory:random") -> Dict[str, object]:
    """The climate frame: `spinup` steps of budgeted observations under each
    rule, then `weeks` weeks free; weekly means scored as eval/memory_climate.py
    (GraphCast-weighted MSE, headline RMSE and ACC against the climatology). A
    run that blows up is scored with the climatology from then on."""
    attach_loss(model)  # the GraphCast weights of the weekly scores
    model.eval()
    if base is not None:
        base.eval()
    targets, keys, kidx, lat_w = _headline(model, dataset)
    horizon = spinup + weeks * week
    runs = [(f"{which}:{rule}", which, rule) for which in (("memory", "n1") if base is not None else ("memory",))
            for rule in (rules if which == "memory" else base_rules)]
    done: Dict[str, List[dict]] = {}
    if progress_path is not None and os.path.exists(progress_path):
        done = torch.load(progress_path, weights_only=False)
    recs = done.setdefault("recs", [])
    weekly = lambda x: x.reshape(weeks, week, *x.shape[1:]).mean(1)
    for n, i in enumerate(indices):
        if n < len(recs):
            continue
        item = dataset[int(i)]
        batch = {k: v[None].to(device) for k, v in item.items()}
        init = int(item["init_time"])
        valid = init + 6 * 3600 * np.arange(spinup + 1, horizon + 1)
        clim = climatology(valid[None])[0].float()
        clim_norm = torch.from_numpy(stats.normalize(clim.numpy(), targets)).float()
        truth = batch["targets"][0, spinup:horizon].float().cpu()
        truth_w, clim_w = weekly(truth), weekly(clim_norm)
        zero_t = torch.zeros(weeks, *truth_w.shape[1:], device=device)
        rec: Dict[str, object] = {"init": init}

        def score(name, pred_w):
            h, a = errors(model, pred_w.to(device), truth_w.to(device), zero_t)
            phys, obs, clim_p = (stats.unnormalize(x, targets) for x in (pred_w, truth_w, clim_w))
            rec[name] = {"all": a.cpu().tolist(),
                         "rmse": weighted_mean((phys - obs).square(), lat_w)[:, kidx].sqrt().tolist(),
                         "acc": acc_per_init(phys[None], obs[None], clim_p[None], lat_w)[0][:, kidx].tolist()}

        score("climatology", clim_w)
        for name, which, rule in runs:
            m = model if which == "memory" else base
            sums = torch.zeros(weeks, *truth.shape[1:])
            blown = [False]

            def on_step(k, pred, dead):
                if k < spinup:
                    return
                blown[0] = blown[0] or bool(dead[0])
                j = k - spinup
                sums[j // week] += clim_norm[j] if blown[0] else pred[0].float().cpu()

            _, choices, _ = play_observation(m, policy, batch, rule, horizon, spinup, 1, budget,
                                             torch.Generator().manual_seed(seed + 7919 * n), cap,
                                             deterministic=(rule == "policy"), on_step=on_step)
            score(name, sums / week)
            rec[name]["blown"] = blown[0]
            freq = np.zeros(policy.patches.n)
            for c in choices:
                freq[c[0].cpu().numpy()] += 1
            rec[name]["freq"] = (freq / max(len(choices), 1)).tolist()
        recs.append(rec)
        if progress_path is not None:
            torch.save(done, progress_path + ".tmp")
            os.replace(progress_path + ".tmp", progress_path)
        if log is not None:
            log(f"[active climate] {n + 1}/{len(indices)}")
    names = ["climatology"] + [name for name, *_ in runs]
    out: Dict[str, object] = {"weeks": list(range(1, weeks + 1)), "spinup_days": spinup / 4, "budget": budget,
                              "keys": keys, "reference": reference, "inits": [r["init"] for r in recs],
                              "runs": {}, "change": {}}
    all_ = {name: np.array([r[name]["all"] for r in recs]) for name in names}
    for name in names:
        rows = [r[name] for r in recs]
        out["runs"][name] = {"all": all_[name].mean(0).tolist(),
                             "rmse": {k: np.array([r["rmse"] for r in rows])[:, :, j].mean(0).tolist()
                                      for j, k in enumerate(keys)},
                             "acc": {k: np.array([r["acc"] for r in rows])[:, :, j].mean(0).tolist()
                                     for j, k in enumerate(keys)},
                             "blowups": int(sum(r.get("blown", False) for r in rows))}
        if rows and "freq" in rows[0]:
            out["runs"][name]["freq"] = np.mean([r["freq"] for r in rows], 0).tolist()
    if reference in all_:
        ref = all_[reference]
        for name in names:
            if name != reference:
                out["change"][name] = _bootstrap((all_[name] - ref) / ref.mean(0), seed)
    return out
