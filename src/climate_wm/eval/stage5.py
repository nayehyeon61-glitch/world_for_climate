"""Stage 5 evaluation: is the composite group used physically, and does the critic judge reality?

Predictions of the design (doc section 7) tested here on validation pairs:

  P2  generator usage: mixing activity in statically unstable columns
      (N^2 < 0 in the lowest layers of the observed state) against stable
      ones, and how the fibre activity spreads over the shells (stage 3's E2
      put 95% of its physics into the boundary-layer shell)
  P3  critic-weighted ensemble: CRPS and spread/skill of softmax(-score / tau)
      weighted members against equal weights, and whether the critic score
      ranks members by their actual error
  P4  counterfactual TOA solar radiation (set to night): latent effect where
      the radiation generator is strong against elsewhere, decoded 2 m
      temperature change (stage 3 E2: 3.7e-3 vs 9e-5, -0.002)
  P5  cross-year invariance: the statistics saved here (Karcher mean and
      log-covariance of the coarse rotations, coefficient moments) are
      compared between years by scripts/report_stage5.py

plus family knock-outs (necessity of each generator family) and the Cartan
split of the coarse rotations into spin (local vorticity) and transport.
P1 (72 h RMSE of near-surface variables) comes from scripts/compare_graphcast.py.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..data.graphcast_mapping import Channel
from ..geometry import composite_group as cg
from ..models.critic import CriticConfig, RealityCritic, member_weights
from ..models.latent import Latent
from ..models.world_model import ClimateWorldModel
from ..train.losses import normalized_latitude_weights
from .physics_diag import DiagnosticsConfig, PhysicsDiagnostics

FAMILY_KEYS = cg.FIB_FAMILIES + ("shift",)
REPORT_VARS = ("geopotential@500", "temperature@850", "specific_humidity@700", "2m_temperature")


def _batches(dataset, indices: Sequence[int], batch_size: int, device):
    indices = list(indices)
    for i in range(0, len(indices), batch_size):
        chunk = [dataset[int(j)] for j in indices[i:i + batch_size]]
        yield {k: torch.utils.data.default_collate([c[k] for c in chunk]).to(device)
               for k in chunk[0]}


def _per_sample_loss(model: ClimateWorldModel, pred: Tensor, target: Tensor) -> Tensor:
    return torch.stack([model.loss_fn(pred[j:j + 1], target[j:j + 1]) for j in range(len(pred))])


def _shell_activity(fib: cg.FibreGenerators, coef: cg.FibreCoefficients,
                    shift: Optional[Tensor]) -> Dict[str, Tensor]:
    """Per family a non-negative activity [B, S, N] (shift: [B, N])."""
    out = {}
    if coef.radiation is not None:
        out["radiation"] = coef.radiation.abs()
    if coef.condensation is not None:
        out["condensation"] = coef.condensation
    if coef.mixing is not None:  # diagonal of the mixing generator per shell
        q2 = fib.modes.square() * fib.mode_eig  # [S, I]
        out["mixing"] = torch.einsum("bni,si->bsn", coef.mixing, q2)
    if coef.surface is not None:
        s = fib.num_shells
        a = coef.surface.abs()[:, None]
        out["surface"] = torch.cat([a, a, a.new_zeros(a.shape[0], s - 2, a.shape[-1])], 1)
    if shift is not None:
        out["shift"] = shift.abs()
    return out


def _unstable_columns(model: ClimateWorldModel, diag: PhysicsDiagnostics, dyn: Tensor,
                      layers: int = 3) -> Tensor:
    """Latent nodes [B, N] whose column has N^2 < 0 in one of the lowest `layers` level pairs."""
    raw = diag.raw(diag.physical(dyn.float()))
    pairs = [lo for lo, _ in diag.pairs[:layers]]
    unstable = torch.stack([raw[("stability", lo)] < 0 for lo in pairs], 1).any(1).float()
    frac = model.graphs.grid_to_latent_mean(unstable[:, None])[..., 0]
    return frac > 0.5


@torch.no_grad()
def composite_usage(model: ClimateWorldModel, dataset, indices: Sequence[int], device="cpu",
                    batch_size: int = 4, tisr_night: Optional[float] = None,
                    log: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """One-step generator statistics, knock-outs and the TISR counterfactual."""
    model.eval()
    lie, layout = model.lie, model.layout
    if lie is None or getattr(lie, "composite", None) is None:
        return {}
    diag = PhysicsDiagnostics(layout, dataset.lat, dataset.lon, DiagnosticsConfig(enabled=True)).to(device)
    diag.set_stats(dataset.stats)
    tisr = [i for i, c in enumerate(layout.forcings) if c.name == "toa_incident_solar_radiation"]
    t2m = [i for i, c in enumerate(layout.targets) if c.name == "2m_temperature"]
    acc: Dict[str, List[Tensor]] = {}
    add = lambda k, v: acc.setdefault(k, []).append(v.detach().float().cpu())
    for n_b, batch in enumerate(_batches(dataset, indices, batch_size, device)):
        hist = model.encode_history(batch)
        nf = model.node_forcing(batch["forcings_tgt"][:, 0])
        target = batch["targets"][:, 0]
        omega, omega_node, _, terms = lie.generate(model.graphs, hist, nf)
        z = lie.act(model.graphs, hist[-1], omega_node, terms)
        pred = model.decode(z)
        loss = _per_sample_loss(model, pred, target)
        add("loss", loss)
        activity = _shell_activity(lie.fibre, terms.coef, terms.shift)
        for k, v in activity.items():
            add(f"act_{k}", v if v.ndim == 3 else v[:, None])
        unstable = _unstable_columns(model, diag, batch["inputs"][:, -1])
        add("unstable", unstable)
        omega0 = lie.split_parts(model.graphs, omega)[0]  # [B, S, 12, 3]
        add("omega0", omega0)
        centres = model.graphs.pyramid.pyramid[-1].patches[0].centres
        k_part, p_part = cg.cartan_split(omega0, centres)
        add("spin2", k_part.square().sum(-1).mean(-1))  # [B, S]
        add("transport2", p_part.square().sum(-1).mean(-1))
        for name, val in terms.coef.active().items():
            add(f"coef_{name}", val.reshape(val.shape[0], -1).mean(-1))
            add(f"coefsq_{name}", val.reshape(val.shape[0], -1).square().mean(-1))
        # knock-outs: one family (or the shift) set to the identity
        for fam in FAMILY_KEYS:
            if fam == "shift":
                if terms.shift is None:
                    continue
                ko = type(terms)(torch.zeros_like(terms.shift), terms.coef)
            else:
                if getattr(terms.coef, fam) is None:
                    continue
                ko = type(terms)(terms.shift, terms.coef.knockout([fam]))
            pred_ko = model.decode(lie.act(model.graphs, hist[-1], omega_node, ko))
            add(f"ko_{fam}", _per_sample_loss(model, pred_ko, target))
        # counterfactual: TOA solar radiation of the target time set to night
        if tisr and tisr_night is not None:
            nf_cf = nf.clone()
            nf_cf[..., tisr[0]] = tisr_night
            _, on_cf, _, t_cf = lie.generate(model.graphs, hist, nf_cf)
            z_cf = lie.act(model.graphs, hist[-1], on_cf, t_cf)
            add("cf_effect", (z_cf.h - z.h).square().mean((1, 3)) + (z_cf.v - z.v).square().sum(-1).mean((1, 3)))
            rad = activity.get("radiation")
            if rad is not None:
                add("cf_rad", rad.sum(1))
            if t2m:
                add("cf_t2m", (model.decode(z_cf) - pred)[:, t2m[0]].mean((-2, -1)))
        if log and n_b % 10 == 0:
            log(f"[usage] {min((n_b + 1) * batch_size, len(indices))}/{len(indices)}")
    cat = {k: torch.cat(v) for k, v in acc.items()}
    shells = [s.name for s in layout.shells.shells]
    out: Dict[str, object] = {"pairs": int(cat["loss"].shape[0]), "loss": float(cat["loss"].mean()),
                              "shells": shells, "families": {}, "knockout": {}}
    unstable = cat["unstable"].bool()
    out["unstable_fraction"] = float(unstable.float().mean())
    for fam in FAMILY_KEYS:
        if f"act_{fam}" not in cat:
            continue
        a = cat[f"act_{fam}"]  # [P, S', N]
        per_shell = a.mean((0, 2))
        share = per_shell / per_shell.sum().clamp_min(1e-12)
        col = a.sum(1)  # [P, N] column activity
        entry = {"mean": float(a.mean()), "per_shell": per_shell.tolist(), "share_per_shell": share.tolist(),
                 "top_shell_share": float(share.max())}
        if unstable.any() and (~unstable).any():
            u, s = float(col[unstable].mean()), float(col[~unstable].mean())
            entry.update(unstable=u, stable=s, unstable_over_stable=u / max(s, 1e-12))
        out["families"][fam] = entry
    fib_total = sum(cat[f"act_{f}"].mean((0, 2)) for f in cg.FIB_FAMILIES if f"act_{f}" in cat)
    if torch.is_tensor(fib_total):
        share = fib_total / fib_total.sum().clamp_min(1e-12)
        out["fibre_share_per_shell"] = share.tolist()
        out["fibre_top_shell_share"] = float(share.max())
    for fam in FAMILY_KEYS:
        if f"ko_{fam}" in cat:
            inc = cat[f"ko_{fam}"] - cat["loss"]
            out["knockout"][fam] = {"loss_increase": float(inc.mean()),
                                    "loss_increase_pct": float(100 * (inc / cat["loss"].clamp_min(1e-8)).mean())}
    out["cartan"] = {"spin_energy_per_shell": cat["spin2"].mean(0).tolist(),
                     "transport_energy_per_shell": cat["transport2"].mean(0).tolist()}
    out["coefficients"] = {name[5:]: {"mean": float(cat[name].mean()),
                                      "rms": float(cat["coefsq_" + name[5:]].mean().sqrt())}
                           for name in cat if name.startswith("coef_")}
    if "cf_effect" in cat:
        eff = cat["cf_effect"]  # [P, N]
        cf = {"tisr_night": tisr_night, "effect": float(eff.mean())}
        if "cf_rad" in cat:
            rad = cat["cf_rad"]
            strong = rad >= rad.quantile(0.75)
            cf.update(effect_where_radiation_strong=float(eff[strong].mean()),
                      effect_elsewhere=float(eff[~strong].mean()))
            cf["concentration"] = cf["effect_where_radiation_strong"] / max(cf["effect_elsewhere"], 1e-12)
        if "cf_t2m" in cat:
            cf["decoded_t2m_change"] = float(cat["cf_t2m"].mean())
        out["counterfactual"] = cf
    om = cat["omega0"]  # [P, S, 12, 3]
    mu = cg.karcher_mean(om.permute(0, 2, 1, 3).reshape(om.shape[0], -1, 3))  # per (patch, shell)
    out["omega0_group_stats"] = {"karcher_mean": mu.tolist(),
                                 "log_cov": cg.log_cov(om.permute(0, 2, 1, 3).reshape(om.shape[0], -1, 3),
                                                       mu).tolist()}
    return out


@torch.no_grad()
def calibrate_critic(model: ClimateWorldModel, dataset, indices: Sequence[int], device="cpu",
                     batch_size: int = 4) -> RealityCritic:
    """The model's critic, or a diagnostic critic calibrated now on observed frames
    (for arms trained without one: post-hoc weighting control)."""
    if model.critic is not None and int(model.critic.diagnostic.count) > 0:
        return model.critic
    diag = PhysicsDiagnostics(model.layout, dataset.lat, dataset.lon, DiagnosticsConfig(enabled=True))
    diag.set_stats(dataset.stats)
    critic = RealityCritic(diag, len(dataset.lat), len(model.layout.shells), None,
                           CriticConfig(enabled=True, fisher=False, momentum=0.9)).to(device)
    for batch in _batches(dataset, indices, batch_size, device):
        for j in range(batch["inputs"].shape[1]):
            critic.diagnostic.observe(batch["inputs"][:, j])
    return critic


def weighted_crps(ens: Tensor, target: Tensor, w: Tensor) -> Tensor:
    """CRPS of a weighted ensemble, sum_i w_i |x_i - y| - 1/2 sum_ij w_i w_j |x_i - x_j|.
    ens [M, ...], target [...], w [M, ...] (broadcastable, sums to 1 over M)."""
    skill = (w * (ens - target[None]).abs()).sum(0)
    pair = (w[:, None] * w[None, :] * (ens[:, None] - ens[None, :]).abs()).sum((0, 1))
    return skill - 0.5 * pair


@torch.no_grad()
def weighted_ensemble(model: ClimateWorldModel, dataset, indices: Sequence[int], critic: RealityCritic,
                      members: int = 4, lead_steps: int = 12, device="cpu", batch_size: int = 2,
                      tau: float = 1.0, log: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Equal vs critic-weighted ensemble CRPS / RMSE / spread-skill per lead and variable,
    and the rank correlation of the critic score with the member error."""
    model.eval()
    layout, stats = model.layout, dataset.stats
    t2d = layout.target_to_dynamic
    keys = [c.key for c in layout.targets]
    var_idx = {v: keys.index(v) for v in REPORT_VARS if v in keys}
    lat_w = torch.tensor(normalized_latitude_weights(dataset.lat), dtype=torch.float32)
    wmean = lambda x: (x * lat_w[:, None]).mean((-2, -1))  # area mean over [.., H, W]
    sums: Dict[str, Tensor] = {}
    corr, count = [], 0
    for n_b, batch in enumerate(_batches(dataset, indices, batch_size, device)):
        steps = min(lead_steps, batch["targets"].shape[1])
        pred = model.rollout(batch, steps, members=members)  # [M, B, L, C, H, W] normalised
        m, b = pred.shape[:2]
        score = torch.stack([critic.diagnostic.score(pred[:, :, k].flatten(0, 1)[:, t2d]).view(m, b)
                             for k in range(steps)], -1)  # [M, B, L]
        w = member_weights(score, tau)
        phys = stats.unnormalize(pred.cpu(), layout.targets)
        truth = stats.unnormalize(batch["targets"][:, :steps].cpu(), layout.targets)
        for v, c in var_idx.items():
            x, y = phys[:, :, :, c], truth[:, :, c]  # [M, B, L, H, W], [B, L, H, W]
            eq = torch.full_like(w, 1.0 / m).cpu()[..., None, None]
            wc = w.cpu()[..., None, None]
            for tag, ww in (("equal", eq), ("weighted", wc)):
                crps = wmean(weighted_crps(x, y, ww))  # [B, L]
                mean = (ww * x).sum(0)
                mse = wmean((mean - y).square())
                var = wmean((ww * (x - mean[None]).square()).sum(0))
                for name, val in (("crps", crps), ("mse", mse), ("var", var)):
                    key = f"{tag}/{name}/{v}"
                    sums[key] = sums.get(key, 0) + val.sum(0)
            err = wmean((x - y[None]).square())  # [M, B, L] member error
            for bi in range(b):
                for k in range(steps):
                    s_rank = score[:, bi, k].cpu().argsort().argsort().float()
                    e_rank = err[:, bi, k].argsort().argsort().float()
                    if m > 2:
                        corr.append(float(np.corrcoef(s_rank.numpy(), e_rank.numpy())[0, 1]))
        count += b
        if log:
            log(f"[ensemble] {min((n_b + 1) * batch_size, len(indices))}/{len(indices)}")
    lead_h = [6 * (k + 1) for k in range(lead_steps)]
    out: Dict[str, object] = {"members": members, "tau": tau, "inits": count, "lead_h": lead_h, "variables": {}}
    for v in var_idx:
        entry = {}
        for tag in ("equal", "weighted"):
            crps = (sums[f"{tag}/crps/{v}"] / count).tolist()
            rmse = (sums[f"{tag}/mse/{v}"] / count).sqrt()
            spread = (sums[f"{tag}/var/{v}"] / count).sqrt() * np.sqrt((members + 1) / members)
            entry[tag] = {"crps": crps, "ens_mean_rmse": rmse.tolist(),
                          "spread_skill": (spread / rmse.clamp_min(1e-12)).tolist()}
        out["variables"][v] = entry
    corr = [c for c in corr if c == c]
    out["critic_error_rank_corr"] = float(np.mean(corr)) if corr else float("nan")
    return out
