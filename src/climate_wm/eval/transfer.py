"""Learning-to-Theorize evaluation for `lie_program` models.

Three questions, mirroring NEO's protocol (Baek et al. 2026) on weather pairs
(Z_t, Z_{t+6h}):

  self-explainability   how much of a 6 h change does the shortest accurate
                        program recover, against persistence (k = 0) and the
                        reconstruction floor (decoding the target latent)?
  analog transfer       does a program inferred on one pair explain the change
                        of the most similar state at another time better than
                        copying the Euclidean latent difference ("delta copy")
                        or a program from a random pair?
  length OOD            do longer programs explain 12 h pairs although training
                        only saw 6 h pairs with at most K steps?

All functions take a model whose transition is `lie_program`; the continuous
control (codebook_size 0) runs through the same code, which keeps the arms
comparable.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..models.latent import Latent
from ..models.physics_codes import FAMILIES, code_family
from ..models.world_model import ClimateWorldModel


@dataclasses.dataclass
class PairSet:
    """Encoded observation pairs (CPU tensors)."""
    s: Latent                 # Z_t             [N, S, Nodes, ...]
    s_y: Latent               # Z_{t+lead}
    nf: Tensor                # node forcing at the target time
    target: Tensor            # normalised target fields [N, C, H, W]
    times: np.ndarray         # init time in seconds [N]
    lead_hours: float
    ns: Optional[Tensor] = None  # static fields at latent nodes [N, Nodes, St]

    def __len__(self) -> int:
        return self.s.h.shape[0]

    def subset(self, idx) -> "PairSet":
        idx = torch.as_tensor(idx)
        return PairSet(Latent(self.s.h[idx], self.s.v[idx]), Latent(self.s_y.h[idx], self.s_y.v[idx]),
                       self.nf[idx], self.target[idx], self.times[idx.numpy()], self.lead_hours,
                       None if self.ns is None else self.ns[idx])


def _cat(latents: Sequence[Latent]) -> Latent:
    return Latent(torch.cat([z.h for z in latents]), torch.cat([z.v for z in latents]))


def _to(z: Latent, device) -> Latent:
    return Latent(z.h.to(device), z.v.to(device))


@torch.no_grad()
def collect_pairs(model: ClimateWorldModel, dataset, indices: Sequence[int],
                  target_step: int = 0, device="cpu", batch_size: int = 4,
                  log: Optional[Callable[[str], None]] = None) -> PairSet:
    """Encodes (x_t, x_{t + 6h (target_step + 1)}) for the given dataset indices."""
    model.eval()
    s, s_y, nf, ns, tgt, times = [], [], [], [], [], []
    indices = list(indices)
    for i in range(0, len(indices), batch_size):
        chunk = [dataset[int(j)] for j in indices[i:i + batch_size]]
        batch = {k: torch.utils.data.default_collate([c[k] for c in chunk]).to(device) for k in chunk[0]}
        z = model.encode(batch["inputs"][:, -1], batch["forcings_in"][:, -1], batch["static"])
        zy = model.encode_target(batch, target_step)
        s.append(Latent(z.h.cpu(), z.v.cpu()))
        s_y.append(Latent(zy.h.cpu(), zy.v.cpu()))
        nf.append(model.node_forcing(batch["forcings_tgt"][:, target_step]).cpu())
        ns.append(model.graphs.grid_to_latent_mean(batch["static"]).cpu())
        tgt.append(batch["targets"][:, target_step].cpu())
        times.append(batch["init_time"].cpu().numpy())
        if log and (i // batch_size) % 10 == 0:
            log(f"[pairs] {min(i + batch_size, len(indices))}/{len(indices)}")
    step_hours = 6.0 * (target_step + 1)
    return PairSet(_cat(s), _cat(s_y), torch.cat(nf), torch.cat(tgt),
                   np.concatenate(times), step_hours, torch.cat(ns))


def _per_sample_loss(model: ClimateWorldModel, z: Latent, target: Tensor) -> Tensor:
    """GraphCast loss of the decoded latent, one value per sample [B]."""
    pred = model.decode(z)
    return torch.stack([model.loss_fn(pred[b:b + 1], target[b:b + 1]) for b in range(pred.shape[0])])


def _per_sample_latent_error(a: Latent, b: Latent) -> Tensor:
    d = torch.cat([(a.h.float() - b.h.float()).flatten(1), (a.v.float() - b.v.float()).flatten(1)], 1)
    return d.square().mean(1)


@dataclasses.dataclass
class Explained:
    """Per-pair explanation statistics (CPU)."""
    lengths: Tensor          # k* per (pair, shell, patch) [N, S, P]
    program: Tensor          # [K, N, S, P, 3]
    indices: Optional[Tensor]  # [K, N, S, P]
    patch_error: Tensor      # [K + 1, N, S, P]
    loss_by_length: Tensor   # decoded loss of state k, [N, K + 1] (0 = persistence)
    loss_selected: Tensor    # decoded loss of the MDL-assembled state [N]
    loss_floor: Tensor       # decoded loss of the target latent [N]
    latent_by_length: Tensor  # latent MSE of state k [N, K + 1]
    latent_selected: Tensor  # [N]
    phys_indices: Optional[Tensor] = None  # physics code per step [K, N, S, P] (phase 2)
    sub_error: Optional[Tensor] = None     # patch error after the Lie sub-step [K, N, S, P]
    phys_gate: Optional[Tensor] = None     # physics gate per step [K, N, S, P]

    def active_physics(self) -> Optional[Tensor]:
        """[K, N, S, P] mask of steps whose physics code is on (non-identity, gate open)."""
        if self.phys_indices is None:
            return None
        active = self.phys_indices > 0
        return active & (self.phys_gate > 0.5) if self.phys_gate is not None else active


@torch.no_grad()
def explain_all(model: ClimateWorldModel, pairs: PairSet, steps: Optional[int] = None,
                device="cpu", batch_size: int = 4,
                log: Optional[Callable[[str], None]] = None) -> Explained:
    model.eval()
    th = model.theory
    out: Dict[str, List[Tensor]] = {k: [] for k in ("lengths", "program", "indices", "patch_error",
                                                    "loss_by_length", "loss_selected", "loss_floor",
                                                    "latent_by_length", "latent_selected",
                                                    "phys_indices", "sub_error", "phys_gate")}
    n = len(pairs)
    for i in range(0, n, batch_size):
        sub = pairs.subset(range(i, min(i + batch_size, n)))
        s, s_y = _to(sub.s, device), _to(sub.s_y, device)
        nf, target = sub.nf.to(device), sub.target.to(device)
        ns = None if sub.ns is None else sub.ns.to(device)
        exp = th.explain(model.graphs, s, s_y, nf, steps=steps, ns=ns)
        out["lengths"].append(exp.lengths.cpu())
        out["program"].append(exp.program.cpu())
        if exp.indices is not None:
            out["indices"].append(exp.indices.cpu())
        if exp.phys_indices is not None:
            out["phys_indices"].append(exp.phys_indices.cpu())
            out["sub_error"].append(exp.sub_error.cpu())
            out["phys_gate"].append(exp.phys_gate.cpu())
        out["patch_error"].append(exp.patch_error.cpu())
        out["loss_by_length"].append(torch.stack(
            [_per_sample_loss(model, z, target) for z in exp.states], 1).cpu())
        out["loss_selected"].append(_per_sample_loss(model, exp.selected, target).cpu())
        out["loss_floor"].append(_per_sample_loss(model, s_y, target).cpu())
        out["latent_by_length"].append(torch.stack(
            [_per_sample_latent_error(z, s_y) for z in exp.states], 1).cpu())
        out["latent_selected"].append(_per_sample_latent_error(exp.selected, s_y).cpu())
        if log and (i // batch_size) % 10 == 0:
            log(f"[explain] {min(i + batch_size, n)}/{n}")
    cat = lambda k, dim=0: torch.cat(out[k], dim)
    return Explained(cat("lengths"), cat("program", 1),
                     cat("indices", 1) if out["indices"] else None,
                     cat("patch_error", 1), cat("loss_by_length"), cat("loss_selected"),
                     cat("loss_floor"), cat("latent_by_length"), cat("latent_selected"),
                     cat("phys_indices", 1) if out["phys_indices"] else None,
                     cat("sub_error", 1) if out["sub_error"] else None,
                     cat("phys_gate", 1) if out["phys_gate"] else None)


def _usage_by_region(active: Tensor, patch_lat: Optional[Tensor]) -> Dict[str, float]:
    """Fraction of active (non-identity) codes per shell and, with patch
    latitudes (degrees, [P]), in the tropics (|lat| < 23.5) vs outside."""
    out = {"per_shell": [float(v) for v in active.float().mean((0, 1, 3))]}
    if patch_lat is not None:
        tropics = patch_lat.abs() < 23.5
        if tropics.any() and (~tropics).any():
            out["tropics"] = float(active[..., tropics].float().mean())
            out["extratropics"] = float(active[..., ~tropics].float().mean())
    return out


def self_explainability(exp: Explained, codebook_size: int,
                        patch_lat: Optional[Tensor] = None) -> Dict[str, object]:
    """Means over pairs; `explained_fraction` = (persistence - selected) / (persistence - floor)."""
    k = exp.loss_by_length.shape[1] - 1
    pers, sel, floor = exp.loss_by_length[:, 0], exp.loss_selected, exp.loss_floor
    frac = ((pers - sel) / (pers - floor).clamp_min(1e-8)).clamp(-5, 5)
    res: Dict[str, object] = {
        "pairs": int(len(sel)),
        "loss_persistence": float(pers.mean()),
        "loss_by_length": [float(v) for v in exp.loss_by_length.mean(0)],
        "loss_selected": float(sel.mean()),
        "loss_floor": float(floor.mean()),
        "explained_fraction": float(frac.mean()),
        "latent_by_length": [float(v) for v in exp.latent_by_length.mean(0)],
        "latent_selected": float(exp.latent_selected.mean()),
        "mean_length": float(exp.lengths.float().mean()),
        "mean_length_per_shell": [float(v) for v in exp.lengths.float().mean((0, 2))],
        "length_histogram": [float(v) for v in torch.bincount(exp.lengths.flatten(), minlength=k + 1).float()
                             / exp.lengths.numel()],
    }
    # does activity (size of the 6 h change) predict program length?
    activity = exp.patch_error[0].flatten().float()
    length = exp.lengths.flatten().float()
    if activity.std() > 0 and length.std() > 0:
        res["corr_length_activity"] = float(torch.corrcoef(torch.stack([activity, length]))[0, 1])
    if exp.indices is not None and codebook_size > 0:
        counts = torch.bincount(exp.indices.flatten(), minlength=codebook_size).float()
        p = counts / counts.sum()
        res["code_usage"] = [float(v) for v in p]
        res["codes_used"] = int((counts > 0).sum())
        res["perplexity"] = float(torch.exp(-(p * torch.log(p + 1e-9)).sum()))
        # per-shell usage [S, M]
        s_count = torch.stack([torch.bincount(exp.indices[:, :, sh].flatten(), minlength=codebook_size)
                               for sh in range(exp.indices.shape[2])]).float()
        res["code_usage_per_shell"] = (s_count / s_count.sum(1, keepdim=True).clamp_min(1)).tolist()
        res["lie_usage"] = _usage_by_region(exp.indices > 0, patch_lat)
    if exp.phys_indices is not None:
        active = exp.active_physics()
        res["phys_frac"] = float(active.float().mean())
        res["phys_codes_used"] = int(len(torch.unique(exp.phys_indices)))
        res["phys_usage"] = _usage_by_region(active, patch_lat)
        if exp.phys_gate is not None:
            res["phys_gate_mean"] = float(exp.phys_gate.mean())
        lie_gain = (exp.patch_error[:-1] - exp.sub_error).sum()
        phys_gain = (exp.sub_error - exp.patch_error[1:]).sum()
        total = float(lie_gain + phys_gain)
        res["phys_share"] = float(phys_gain / total) if abs(total) > 1e-12 else 0.0
    return res


def find_analogs(pairs: PairSet, min_gap_hours: float = 168.0) -> Tensor:
    """For each pair i the index j of the nearest state Z_t (full latent L2) at
    least `min_gap_hours` away in time; -1 when none qualifies."""
    x = torch.cat([pairs.s.h.flatten(1), pairs.s.v.flatten(1)], 1).float()
    d = torch.cdist(x, x)
    t = torch.as_tensor(pairs.times, dtype=torch.float64)
    gap = (t[:, None] - t[None]).abs() / 3600.0
    d[gap < min_gap_hours] = float("inf")
    best = d.argmin(1)
    best[torch.isinf(d.min(1).values)] = -1
    return best


@torch.no_grad()
def analog_transfer(model: ClimateWorldModel, pairs: PairSet, exp: Explained,
                    analogs: Tensor, device="cpu", batch_size: int = 4,
                    seed: int = 0) -> Dict[str, object]:
    """Executes pair i's program on its analog state j and scores it against
    the analog's own future, next to persistence, delta copy, a program from a
    random pair and the analog's own explanation (oracle)."""
    model.eval()
    th, graphs = model.theory, model.graphs
    pos = graphs.patches.node_pos
    valid = torch.nonzero(analogs >= 0).flatten()
    gen = torch.Generator().manual_seed(seed)
    rand = torch.randint(0, len(pairs), (len(valid),), generator=gen)
    rows = {k: [] for k in ("transfer", "persistence", "delta", "random", "oracle",
                            "transfer_latent", "persistence_latent", "delta_latent",
                            "random_latent", "oracle_latent")}
    for b in range(0, len(valid), batch_size):
        ii = valid[b:b + batch_size]
        jj = analogs[ii]
        rr = rand[b:b + batch_size]
        src, ana = pairs.subset(ii), pairs.subset(jj)
        s_j, sy_j = _to(ana.s, device), _to(ana.s_y, device)
        target_j = ana.target.to(device)
        nf_j = ana.nf.to(device)
        ns_j = None if ana.ns is None else ana.ns.to(device)
        phys = lambda kk: None if exp.phys_indices is None else exp.phys_indices[:, kk].to(device)
        gate = lambda kk: None if exp.phys_gate is None else exp.phys_gate[:, kk].to(device)
        run = lambda kk: th.execute(graphs, s_j, exp.program[:, kk].to(device),
                                    exp.lengths[kk].to(device), phys(kk), nf_j, ns_j, gate(kk))
        # (1) transfer: program inferred on i, executed on the analog state j
        z_tr = run(ii)
        # (2) delta copy: Euclidean latent change of i added to j (re-projected)
        delta = Latent(src.s_y.h - src.s.h, src.s_y.v - src.s.v)
        z_delta = s_j.exp(_to(delta, device), pos)
        # (3) a program from a random pair
        z_rand = run(rr)
        # (4) oracle: j explained with its own target
        z_or = run(jj)
        for name, z in (("transfer", z_tr), ("persistence", s_j), ("delta", z_delta),
                        ("random", z_rand), ("oracle", z_or)):
            rows[name].append(_per_sample_loss(model, z, target_j).cpu())
            rows[name + "_latent"].append(_per_sample_latent_error(z, sy_j).cpu())
    out: Dict[str, object] = {"pairs": int(len(valid))}
    if not len(valid):
        return out
    val = {k: torch.cat(v) for k, v in rows.items()}
    for k, v in val.items():
        out["loss_" + k if not k.endswith("_latent") else k] = float(v.mean())
    pers, tr, dl, orc = val["persistence"], val["transfer"], val["delta"], val["oracle"]
    out["win_vs_delta"] = float((tr < dl).float().mean())
    out["win_vs_persistence"] = float((tr < pers).float().mean())
    out["win_vs_random"] = float((tr < val["random"]).float().mean())
    out["transfer_gain"] = float((((pers - tr) / (pers - orc).clamp_min(1e-8)).clamp(-5, 5)).mean())
    out["delta_gain"] = float((((pers - dl) / (pers - orc).clamp_min(1e-8)).clamp(-5, 5)).mean())
    return out


@torch.no_grad()
def length_ood(model: ClimateWorldModel, pairs_long: PairSet, steps: Sequence[int],
               device="cpu", batch_size: int = 4,
               log: Optional[Callable[[str], None]] = None) -> Dict[str, object]:
    """Explains longer-lead pairs with programs of several maximal lengths."""
    out: Dict[str, object] = {"lead_hours": pairs_long.lead_hours, "by_steps": {}}
    for k in steps:
        exp = explain_all(model, pairs_long, steps=k, device=device, batch_size=batch_size, log=log)
        out["by_steps"][str(k)] = {
            "loss_persistence": float(exp.loss_by_length[:, 0].mean()),
            "loss_selected": float(exp.loss_selected.mean()),
            "loss_full": float(exp.loss_by_length[:, -1].mean()),
            "loss_floor": float(exp.loss_floor.mean()),
            "latent_selected": float(exp.latent_selected.mean()),
            "mean_length": float(exp.lengths.float().mean()),
            "explained_fraction": float((((exp.loss_by_length[:, 0] - exp.loss_selected)
                                          / (exp.loss_by_length[:, 0] - exp.loss_floor).clamp_min(1e-8))
                                         .clamp(-5, 5)).mean()),
        }
    return out


def _family_used(exp: Explained, family: int, num_codes: int) -> Tensor:
    """[N, S, P] mask: the selected program of the patch uses a code of `family`."""
    k = torch.arange(exp.phys_indices.shape[0])[:, None, None, None]
    active = (k < exp.lengths[None]) & exp.active_physics() \
        & (code_family(exp.phys_indices, num_codes) == family)
    return active.any(0)


def _by_region(x: Tensor, patch_lat: Optional[Tensor]) -> Dict[str, object]:
    """Means of a [N, S, P] field: overall, per shell and tropics vs extratropics."""
    out = {"all": float(x.mean()), "per_shell": [float(v) for v in x.mean((0, 2))]}
    if patch_lat is not None:
        tropics = patch_lat.abs() < 23.5
        if tropics.any() and (~tropics).any():
            out["tropics"] = float(x[..., tropics].mean())
            out["extratropics"] = float(x[..., ~tropics].mean())
    return out


@torch.no_grad()
def causal_tests(model: ClimateWorldModel, pairs: PairSet, exp: Explained,
                 patch_lat: Optional[Tensor], forcing_channel: int, forcing_value: float,
                 report_channels: Dict[str, int], device="cpu", batch_size: int = 4
                 ) -> Dict[str, object]:
    """Intervention tests on the inferred programs (phase 2 models only).

    counterfactual   re-executes every pair's own program with forcing channel
                     `forcing_channel` set to `forcing_value` (e.g. TOA solar
                     radiation = night); only codes that read the forcing can
                     respond, so the effect should sit on patches that used
                     radiation or surface codes, and the decoded change of the
                     channels in `report_channels` (e.g. 2 m temperature) gives
                     the sign of the learned response.
    knockout         sets one physics family to the identity and measures how
                     much the explanation error grows (necessity of the family),
                     overall, per shell and tropics vs extratropics.
    """
    th, graphs = model.theory, model.graphs
    if th.physics is None or exp.phys_indices is None:
        return {}
    n = len(pairs)
    fact_loss, cf_effect, cf_chan, ko_loss = [], [], {}, {f: [] for f in range(len(FAMILIES))}
    ko_err = {f: [] for f in range(len(FAMILIES))}
    fact_err = []
    for b in range(0, n, batch_size):
        ii = torch.arange(b, min(b + batch_size, n))
        sub = pairs.subset(ii)
        s, s_y = _to(sub.s, device), _to(sub.s_y, device)
        nf, target = sub.nf.to(device), sub.target.to(device)
        ns = None if sub.ns is None else sub.ns.to(device)
        prog, lens = exp.program[:, ii].to(device), exp.lengths[ii].to(device)
        phys = exp.phys_indices[:, ii].to(device)
        gates = None if exp.phys_gate is None else exp.phys_gate[:, ii].to(device)
        z_fact = th.execute(graphs, s, prog, lens, phys, nf, ns, gates)
        pred_fact = model.decode(z_fact)
        fact_loss.append(torch.stack([model.loss_fn(pred_fact[j:j + 1], target[j:j + 1])
                                      for j in range(len(ii))]).cpu())
        fact_err.append(th._patch_error(graphs, z_fact, s_y).cpu())
        # counterfactual forcing
        nf_cf = nf.clone()
        nf_cf[..., forcing_channel] = forcing_value
        z_cf = th.execute(graphs, s, prog, lens, phys, nf_cf, ns, gates)
        cf_effect.append(th._patch_error(graphs, z_cf, z_fact).cpu())
        diff = (model.decode(z_cf) - pred_fact).mean((-2, -1)).cpu()  # [B, C]
        for name, c in report_channels.items():
            cf_chan.setdefault(name, []).append(diff[:, c])
        # family knock-outs
        fam = code_family(phys, th.cfg.physics_codes)
        for f in range(len(FAMILIES)):
            z_ko = th.execute(graphs, s, prog, lens, torch.where(fam == f, torch.zeros_like(phys), phys),
                              nf, ns, gates)
            pred_ko = model.decode(z_ko)
            ko_loss[f].append(torch.stack([model.loss_fn(pred_ko[j:j + 1], target[j:j + 1])
                                           for j in range(len(ii))]).cpu())
            ko_err[f].append(th._patch_error(graphs, z_ko, s_y).cpu())
    fact_loss, fact_err = torch.cat(fact_loss), torch.cat(fact_err)
    effect = torch.cat(cf_effect)  # [N, S, P] latent effect of the intervention
    out: Dict[str, object] = {
        "loss_factual_replay": float(fact_loss.mean()),
        "counterfactual": {
            "forcing_channel": forcing_channel, "forcing_value": forcing_value,
            "effect": _by_region(effect, patch_lat),
            "effect_where_family_used": {},
            "decoded_channel_change": {k: float(torch.cat(v).mean()) for k, v in cf_chan.items()},
        },
        "knockout": {},
    }
    any_phys = torch.zeros_like(effect, dtype=torch.bool)
    for f, name in enumerate(FAMILIES):
        used = _family_used(exp, f, th.cfg.physics_codes)
        any_phys |= used
        if used.any():
            out["counterfactual"]["effect_where_family_used"][name] = float(effect[used].mean())
    if (~any_phys).any():
        out["counterfactual"]["effect_where_family_used"]["none"] = float(effect[~any_phys].mean())
    for f, name in enumerate(FAMILIES):
        loss_ko = torch.cat(ko_loss[f])
        err_ko = torch.cat(ko_err[f])
        out["knockout"][name] = {
            "loss_increase": float((loss_ko - fact_loss).mean()),
            "loss_increase_pct": float(100 * ((loss_ko - fact_loss) / fact_loss.clamp_min(1e-8)).mean()),
            "patch_error_increase": _by_region(err_ko - fact_err, patch_lat),
        }
    return out


def markdown(results: Dict[str, object]) -> str:
    se, tr = results["self_explainability"], results.get("analog_transfer", {})
    lines = [f"**{results.get('experiment', '')}** — {se['pairs']} pairs at {results['lead_hours']:.0f} h",
             "", "| quantity | value |", "|---|---|",
             f"| loss: persistence (k=0) | {se['loss_persistence']:.4f} |",
             f"| loss: MDL-selected | {se['loss_selected']:.4f} |",
             f"| loss: full length K | {se['loss_by_length'][-1]:.4f} |",
             f"| loss: reconstruction floor | {se['loss_floor']:.4f} |",
             f"| explained fraction of the change | {se['explained_fraction']:.3f} |",
             f"| mean program length k* | {se['mean_length']:.2f} |"]
    if "codes_used" in se:
        lines.append(f"| codes used / perplexity | {se['codes_used']} / {se['perplexity']:.2f} |")
    if "corr_length_activity" in se:
        lines.append(f"| corr(k*, activity) | {se['corr_length_activity']:.3f} |")
    if "phys_frac" in se:
        pu, lu = se["phys_usage"], se.get("lie_usage", {})
        lines.append(f"| physics codes: share of steps / of error reduction | "
                     f"{100 * se['phys_frac']:.0f}% / {100 * se['phys_share']:.0f}% |")
        lines.append("| physics usage per shell (surface → top) | "
                     + ", ".join(f"{100 * v:.0f}%" for v in pu["per_shell"]) + " |")
        if "tropics" in pu:
            lines.append(f"| physics usage tropics / extratropics | "
                         f"{100 * pu['tropics']:.0f}% / {100 * pu['extratropics']:.0f}% |")
        if "tropics" in lu:
            lines.append(f"| Lie (non-identity) usage tropics / extratropics | "
                         f"{100 * lu['tropics']:.0f}% / {100 * lu['extratropics']:.0f}% |")
    if tr.get("pairs"):
        lines += ["", f"Analog transfer ({tr['pairs']} pairs):", "",
                  "| state on the analog | decoded loss | latent MSE |", "|---|---|---|",
                  f"| persistence | {tr['loss_persistence']:.4f} | {tr['persistence_latent']:.4f} |",
                  f"| delta copy | {tr['loss_delta']:.4f} | {tr['delta_latent']:.4f} |",
                  f"| random program | {tr['loss_random']:.4f} | {tr['random_latent']:.4f} |",
                  f"| **transferred program** | **{tr['loss_transfer']:.4f}** | **{tr['transfer_latent']:.4f}** |",
                  f"| oracle (own program) | {tr['loss_oracle']:.4f} | {tr['oracle_latent']:.4f} |",
                  "",
                  f"transfer beats delta copy on {100 * tr['win_vs_delta']:.0f}% of pairs, "
                  f"persistence on {100 * tr['win_vs_persistence']:.0f}%, a random program on "
                  f"{100 * tr['win_vs_random']:.0f}%; gain vs oracle: transfer {tr['transfer_gain']:.3f}, "
                  f"delta {tr['delta_gain']:.3f}"]
    ca = results.get("causal")
    if ca:
        cf, ko = ca["counterfactual"], ca["knockout"]
        lines += ["", f"Intervention tests (factual replay loss {ca['loss_factual_replay']:.4f}):", "",
                  f"- counterfactual forcing (channel {cf['forcing_channel']} := {cf['forcing_value']:.2f}): "
                  f"latent effect {cf['effect']['all']:.2e} overall"
                  + (f", tropics {cf['effect']['tropics']:.2e} / extratropics {cf['effect']['extratropics']:.2e}"
                     if "tropics" in cf["effect"] else "")
                  + "; where a family was used: "
                  + ", ".join(f"{k} {v:.2e}" for k, v in cf["effect_where_family_used"].items())
                  + "; decoded change: "
                  + ", ".join(f"{k} {v:+.4f}" for k, v in cf["decoded_channel_change"].items()),
                  "", "| knocked-out family | loss increase | % | patch error increase: tropics / extratropics | per shell (surface → top) |",
                  "|---|---|---|---|---|"]
        for name, v in ko.items():
            pe = v["patch_error_increase"]
            region = (f"{pe['tropics']:.2e} / {pe['extratropics']:.2e}" if "tropics" in pe else "–")
            lines.append(f"| {name} | {v['loss_increase']:+.4f} | {v['loss_increase_pct']:+.2f}% | {region} | "
                         + ", ".join(f"{x:.1e}" for x in pe["per_shell"]) + " |")
    ood = results.get("length_ood")
    if ood:
        lines += ["", f"Length OOD ({ood['lead_hours']:.0f} h pairs):", "",
                  "| max steps | persistence | selected | full | floor | mean k* |", "|---|---|---|---|---|---|"]
        for k, v in ood["by_steps"].items():
            lines.append(f"| {k} | {v['loss_persistence']:.4f} | {v['loss_selected']:.4f} | "
                         f"{v['loss_full']:.4f} | {v['loss_floor']:.4f} | {v['mean_length']:.2f} |")
    return "\n".join(lines)
