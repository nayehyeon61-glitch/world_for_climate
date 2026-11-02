"""Stage 4 evaluation: so(3) pyramid energy spectrum and per-level analog transfer.

Two of the three predictions of the stage 4 design are tested here on 6 h
pairs of the validation year:

  * energy spectrum: the mean squared rotation per pyramid level
    (omega^(0), delta^(1), delta^(2)) should fall with the level and the
    finest level should be largest in mid-latitudes;
  * per-level transfer: the rotations inferred for pair i are executed on an
    analog state j (nearest state in latent space, >= 7 days away). Mixing
    levels tells which scale generalises: `coarse` takes omega^(0), delta^(1)
    from i and keeps j's own finest detail, `fine` the opposite. Every variant
    is compared with the same mix taken from a random pair, with persistence
    and with j's own rotations (the structural step the model itself makes).

The third prediction (72-120 h skill) uses the stage 1 protocol
(scripts/compare_graphcast.py); scripts/report_pyramid.py joins all three.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from ..models.latent import Latent
from ..models.pyramid_transition import PyramidLieGenerator
from ..models.world_model import ClimateWorldModel

BANDS = (("tropics", 0.0, 30.0), ("midlat", 30.0, 60.0), ("polar", 60.0, 90.1))


@dataclasses.dataclass
class PyramidPairs:
    """Encoded (history, target) pairs on the CPU."""
    hist: List[Latent]  # history frames, each [N, S, nodes, ...]
    s_y: Latent
    nf: Tensor
    target: Tensor
    times: np.ndarray
    lead_hours: float

    def __len__(self) -> int:
        return self.s_y.h.shape[0]

    def subset(self, idx) -> "PyramidPairs":
        idx = torch.as_tensor(idx)
        sub = lambda z: Latent(z.h[idx], z.v[idx])
        return PyramidPairs([sub(z) for z in self.hist], sub(self.s_y), self.nf[idx],
                            self.target[idx], self.times[idx.numpy()], self.lead_hours)


def _cat(latents: Sequence[Latent]) -> Latent:
    return Latent(torch.cat([z.h for z in latents]), torch.cat([z.v for z in latents]))


def _to(z: Latent, device) -> Latent:
    return Latent(z.h.to(device), z.v.to(device))


@torch.no_grad()
def collect_pairs(model: ClimateWorldModel, dataset, indices: Sequence[int], device="cpu",
                  batch_size: int = 4, log: Optional[Callable[[str], None]] = None) -> PyramidPairs:
    model.eval()
    hist, s_y, nf, tgt, times = None, [], [], [], []
    indices = list(indices)
    for i in range(0, len(indices), batch_size):
        chunk = [dataset[int(j)] for j in indices[i:i + batch_size]]
        batch = {k: torch.utils.data.default_collate([c[k] for c in chunk]).to(device) for k in chunk[0]}
        h = model.encode_history(batch)
        if hist is None:
            hist = [[] for _ in h]
        for store, z in zip(hist, h):
            store.append(Latent(z.h.cpu(), z.v.cpu()))
        zy = model.encode_target(batch, 0)
        s_y.append(Latent(zy.h.cpu(), zy.v.cpu()))
        nf.append(model.node_forcing(batch["forcings_tgt"][:, 0]).cpu())
        tgt.append(batch["targets"][:, 0].cpu())
        times.append(batch["init_time"].cpu().numpy())
        if log and (i // batch_size) % 10 == 0:
            log(f"[pairs] {min(i + batch_size, len(indices))}/{len(indices)}")
    return PyramidPairs([_cat(h) for h in hist], _cat(s_y), torch.cat(nf), torch.cat(tgt),
                        np.concatenate(times), 6.0)


@torch.no_grad()
def explain_all(model: ClimateWorldModel, pairs: PyramidPairs, device="cpu",
                batch_size: int = 4, log: Optional[Callable[[str], None]] = None) -> List[Tensor]:
    """The model's own pyramid rotations per pair: [omega^(0), delta^(1), ...],
    each [N, S, P_k, 3] on the CPU."""
    model.eval()
    parts: Optional[List[List[Tensor]]] = None
    for b in range(0, len(pairs), batch_size):
        sub = pairs.subset(range(b, min(b + batch_size, len(pairs))))
        hist = [_to(z, device) for z in sub.hist]
        omega, _, _ = model.lie(model.graphs, hist, sub.nf.to(device))
        pieces = PyramidLieGenerator.split_parts(model.graphs, omega)
        if parts is None:
            parts = [[] for _ in pieces]
        for store, p in zip(parts, pieces):
            store.append(p.cpu())
        if log and (b // batch_size) % 10 == 0:
            log(f"[explain] {min(b + batch_size, len(pairs))}/{len(pairs)}")
    return [torch.cat(p) for p in parts]


def patch_latitudes(model: ClimateWorldModel) -> List[Tensor]:
    """Latitude (deg) of every patch centre per pyramid level."""
    pyr = model.graphs.pyramid
    out = []
    for k in pyr.rot_levels:
        c = pyr.pyramid[-1].patches[k].centres.cpu()
        out.append(torch.rad2deg(torch.asin(c[:, 2].clamp(-1, 1))))
    return out


def level_energy(parts: Sequence[Tensor], lats: Sequence[Tensor],
                 shell_names: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Mean squared rotation per level (rad^2 per step), by latitude band and shell."""
    energies = [float(p.square().sum(-1).mean()) for p in parts]
    total = sum(energies) + 1e-12
    levels = []
    for k, (p, lat) in enumerate(zip(parts, lats)):
        e = p.square().sum(-1)  # [N, S, P]
        row = {"level": k, "patches": int(p.shape[-2]), "energy": energies[k],
               "share": energies[k] / total,
               "rms_deg": float(np.rad2deg(np.sqrt(energies[k])))}
        for name, lo, hi in BANDS:
            m = (lat.abs() >= lo) & (lat.abs() < hi)
            row[name] = float(e[..., m].mean()) if m.any() else float("nan")
        if shell_names:
            row["by_shell"] = {n: float(e[:, s].mean()) for s, n in enumerate(shell_names)}
        levels.append(row)
    fine = levels[-1]
    return {
        "levels": levels,
        "monotone": all(levels[k]["energy"] > levels[k + 1]["energy"] for k in range(len(levels) - 1)),
        "fine_midlat_over_tropics": (fine["midlat"] / fine["tropics"]
                                     if fine["tropics"] == fine["tropics"] and fine["tropics"] > 0
                                     else float("nan")),
    }


def find_analogs(pairs: PyramidPairs, min_gap_hours: float = 168.0) -> Tensor:
    """Nearest state (full stacked latent, L2) at least `min_gap_hours` away; -1 if none."""
    z = pairs.hist[-1]
    x = torch.cat([z.h.flatten(1), z.v.flatten(1)], 1).float()
    d = torch.cdist(x, x)
    t = torch.as_tensor(pairs.times, dtype=torch.float64)
    gap = (t[:, None] - t[None]).abs() / 3600.0
    d[gap < min_gap_hours] = float("inf")
    best = d.argmin(1)
    best[torch.isinf(d.min(1).values)] = -1
    return best


def _per_sample_latent_error(a: Latent, b: Latent) -> Tensor:
    d = torch.cat([(a.h.float() - b.h.float()).flatten(1), (a.v.float() - b.v.float()).flatten(1)], 1)
    return d.square().mean(1)


def _per_level_latent_error(model: ClimateWorldModel, a: Latent, b: Latent) -> Tensor:
    pyr = model.graphs.pyramid
    rows = []
    for h, v, th, tv in zip(pyr.split(a.h), pyr.split(a.v), pyr.split(b.h), pyr.split(b.v)):
        rows.append(_per_sample_latent_error(Latent(h, v), Latent(th, tv)))
    return torch.stack(rows, 1)  # [B, L]


def _per_sample_loss(model: ClimateWorldModel, z: Latent, target: Tensor) -> Tensor:
    pred = model.decode(z)
    return torch.stack([model.loss_fn(pred[b:b + 1], target[b:b + 1]) for b in range(pred.shape[0])])


@torch.no_grad()
def level_transfer(model: ClimateWorldModel, pairs: PyramidPairs, parts: Sequence[Tensor],
                   analogs: Tensor, device="cpu", batch_size: int = 4, seed: int = 0
                   ) -> Dict[str, object]:
    """Executes mixed-level rotations on analog states (see module docstring)."""
    model.eval()
    graphs, lie = model.graphs, model.lie
    k_fine = len(parts) - 1
    coarse = list(range(k_fine))
    valid = torch.nonzero(analogs >= 0).flatten()
    gen = torch.Generator().manual_seed(seed)
    rand = torch.randint(0, len(pairs), (len(valid),), generator=gen)
    variants = ("persistence", "own", "all", "coarse", "fine", "random_all", "random_coarse", "random_fine")
    rows = {v: [] for v in variants}
    rows_latent = {v: [] for v in variants}
    rows_level = {v: [] for v in variants}

    def mix(src_coarse, src_fine):
        return [parts[k][src_coarse if k in coarse else src_fine].to(device) for k in range(len(parts))]

    for b in range(0, len(valid), batch_size):
        ii = valid[b:b + batch_size]
        jj = analogs[ii]
        rr = rand[b:b + batch_size]
        ana = pairs.subset(jj)
        s_j, sy_j = _to(ana.hist[-1], device), _to(ana.s_y, device)
        target_j = ana.target.to(device)
        plans = {"own": mix(jj, jj), "all": mix(ii, ii), "coarse": mix(ii, jj), "fine": mix(jj, ii),
                 "random_all": mix(rr, rr), "random_coarse": mix(rr, jj), "random_fine": mix(jj, rr)}
        states = {"persistence": s_j}
        for name, plan in plans.items():
            omega_node = lie.node_algebra(graphs, plan)
            states[name] = lie.act(graphs, s_j, omega_node)
        for name, z in states.items():
            rows[name].append(_per_sample_loss(model, z, target_j).cpu())
            rows_latent[name].append(_per_sample_latent_error(z, sy_j).cpu())
            rows_level[name].append(_per_level_latent_error(model, z, sy_j).cpu())
    out: Dict[str, object] = {"pairs": int(len(valid)), "coarse_levels": coarse, "fine_level": k_fine}
    if not len(valid):
        return out
    loss = {k: torch.cat(v) for k, v in rows.items()}
    lat = {k: torch.cat(v) for k, v in rows_latent.items()}
    lev = {k: torch.cat(v) for k, v in rows_level.items()}
    pers, own = lat["persistence"], lat["own"]
    gain = lambda x: float((((pers - x) / (pers - own).clamp_min(1e-8)).clamp(-5, 5)).mean())
    for name in variants:
        out["loss_" + name] = float(loss[name].mean())
        out["latent_" + name] = float(lat[name].mean())
        out["levels_" + name] = [float(x) for x in lev[name].mean(0)]
        if name not in ("persistence", "own"):
            out["gain_" + name] = gain(lat[name])
            out["win_" + name] = float((lat[name] < pers).float().mean())
    out["gain_own"] = 1.0
    out["explained_fraction_own"] = float(((pers - own) / pers.clamp_min(1e-8)).mean())
    # the two numbers prediction 1 is about: how much more a *matched* coarse /
    # fine level helps than a random one, in units of the persistence -> own gap
    out["coarse_gap"] = out["gain_coarse"] - out["gain_random_coarse"]
    out["fine_gap"] = out["gain_fine"] - out["gain_random_fine"]
    out["all_gap"] = out["gain_all"] - out["gain_random_all"]
    return out


def markdown(results: Dict[str, object]) -> str:
    lines = [f"# so(3) pyramid evaluation: {results.get('experiment', '')}", ""]
    en = results.get("energy")
    if en:
        lines += ["## Rotation energy per level (rad^2 per 6 h step)", "",
                  "| level | patches | energy | share | rms (deg) | tropics | midlat | polar |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in en["levels"]:
            lines.append(f"| {r['level']} | {r['patches']} | {r['energy']:.3e} | {r['share']:.2f} | "
                         f"{r['rms_deg']:.2f} | {r['tropics']:.3e} | {r['midlat']:.3e} | {r['polar']:.3e} |")
        lines += ["", f"monotone (coarse > fine): {en['monotone']}; finest level midlat / tropics = "
                      f"{en['fine_midlat_over_tropics']:.2f}", ""]
    tr = results.get("transfer")
    if tr and tr.get("pairs"):
        lines += [f"## Per-level analog transfer ({tr['pairs']} pairs)", "",
                  "| variant | latent MSE | decoded loss | gain | win vs persistence |", "|---|---|---|---|---|"]
        for name in ("persistence", "own", "all", "random_all", "coarse", "random_coarse", "fine", "random_fine"):
            g = tr.get("gain_" + name, float("nan"))
            w = tr.get("win_" + name, float("nan"))
            lines.append(f"| {name} | {tr['latent_' + name]:.4f} | {tr['loss_' + name]:.4f} | "
                         f"{g:.3f} | {w:.2f} |")
        lines += ["", f"explained fraction (own rotations): {tr['explained_fraction_own']:.3f}; "
                      f"matched-minus-random gain: all {tr['all_gap']:.3f}, coarse {tr['coarse_gap']:.3f}, "
                      f"fine {tr['fine_gap']:.3f}", ""]
    return "\n".join(lines)
