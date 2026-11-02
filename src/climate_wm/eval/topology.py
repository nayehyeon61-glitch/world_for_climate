"""Stage 8 phase 1: is there a transition-cost topology in the atmosphere's states?

The design (Korean): "8단계 설계: 동역학적 위상(전이 비용) 학습". The states
are 6 h ERA5 frames seen through the frozen world model; distances under
several representations are compared with the transition structure of the
data itself.

Representations (each maps a state to a point, distances are Euclidean):
    raw_z    pooled latent invariants of the frozen encoder (shell x patch
             scalar means and vector norms), deseasonalised and standardised
    energy   dry total energy norm of the state anomaly (u, v, T on five
             levels and mean sea level pressure, 2x coarsened grid)
    psi_*    commute-time heads (models/commute_time.py) on raw_z

Tests on the held-out year (pairs and clusters never seen by the heads):
    T1  does distance reflect transition difficulty? Spearman between the
        distance of cluster centroids and the commute time of a Markov state
        model (k-means clusters, all years' 6 h transitions); and the growth
        of distance with time lag
    T2  new geometry or new topology? Geometry: slope of log variance of the
        representation's principal directions against log Laplacian
        eigenvalue (CTWM predicts -1). Topology: H0 persistence (single
        linkage merge heights) above a Gaussian null with the same
        covariance, and the residence time in k-means regimes
    T3  do cost neighbourhoods predict better? Analog forecasts (mean of the
        library's k nearest states' successors) for z500, t850, t2m at
        24-120 h, overall and per regime (Atlantic-European blocking, jet
        latitude); and whether local speed / rarity tracks the world model's
        own 72 h error
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from scipy.cluster.hierarchy import linkage
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

from ..data.graphcast_mapping import Channel
from ..geometry.shells import SpherePatches
from ..train.losses import normalized_latitude_weights

ENERGY_LEVELS = (200, 300, 500, 700, 850)
ANALOG_KEYS = ("geopotential@500", "temperature@850", "2m_temperature")
CP, T_REF, R_D, P_REF = 1004.0, 270.0, 287.0, 1.0e5
STEP_H = 6


# ------------------------------------------------------------------ states
def latent_invariants(z, patches: SpherePatches) -> torch.Tensor:
    """Rotation-invariant patch means [B, S, P, cs + cv] of a latent state."""
    feats = torch.cat([z.h, z.v.norm(dim=-1)], -1)  # [B, S, N, c]
    b, s, n, c = feats.shape
    pooled = patches.pool(feats.permute(0, 2, 1, 3).reshape(b, n, s * c))
    return pooled.reshape(b, -1, s, c).permute(0, 2, 1, 3)


def energy_weights(channels: Sequence[Channel], lat: np.ndarray):
    """Channel indices and weights whose squared sum is the dry total energy
    0.5 (u^2 + v^2) + 0.5 cp/T_r T^2 + 0.5 R T_r (p/p_r)^2, per area (sqrt cos lat)."""
    keys = [c.key for c in channels]
    idx, w = [], []
    for lev in ENERGY_LEVELS:
        for name, weight in (("u_component_of_wind", np.sqrt(0.5)), ("v_component_of_wind", np.sqrt(0.5)),
                             ("temperature", np.sqrt(0.5 * CP / T_REF))):
            key = f"{name}@{lev}"
            if key in keys:
                idx.append(keys.index(key))
                w.append(weight)
    if "mean_sea_level_pressure" in keys:
        idx.append(keys.index("mean_sea_level_pressure"))
        w.append(np.sqrt(0.5 * R_D * T_REF) / P_REF)
    area = np.sqrt(np.cos(np.deg2rad(np.asarray(lat, np.float64)))).clip(0.05)
    return np.asarray(idx), np.asarray(w, np.float32), area.astype(np.float32)


def coarsen(x: np.ndarray, f: int = 2) -> np.ndarray:
    """f x f block mean of [..., H, W] (a trailing odd row or column is dropped)."""
    h, w = (d - d % f for d in x.shape[-2:])
    x = x[..., :h, :w]
    return x.reshape(*x.shape[:-2], h // f, f, w // f, f).mean((-3, -1))


@torch.no_grad()
def encode_states(model, dataset, stats, patches: SpherePatches, batch_size: int = 4,
                  log=None) -> Dict[str, np.ndarray]:
    """Per window k of `dataset` (target_steps >= 1): latent invariants of X_t,
    the forcing of the transition t -> t+1 (patch-mean incoming solar
    radiation at t+1 and the season), analog fields and energy features of X_t."""
    layout = model.layout
    dyn = list(layout.dynamic)
    keys = [c.key for c in dyn]
    field_idx = [keys.index(k) for k in ANALOG_KEYS]
    lat_c = coarsen(np.asarray(dataset.lat, np.float32)[:, None].repeat(2, 1))[:, 0]  # the 2x coarsened grid
    e_idx, e_w, area_c = energy_weights(dyn, lat_c)
    forc_keys = [c.name for c in layout.forcings]
    tisr = forc_keys.index("toa_incident_solar_radiation") if "toa_incident_solar_radiation" in forc_keys else None
    season = [forc_keys.index(k) for k in ("year_progress_sin", "year_progress_cos") if k in forc_keys]
    out: Dict[str, List[np.ndarray]] = {k: [] for k in ("latent", "action", "fields", "energy", "times")}
    n = len(dataset)
    for start in range(0, n, batch_size):
        items = [dataset[k] for k in range(start, min(start + batch_size, n))]
        batch = torch.utils.data.default_collate(items)
        x, f, st = batch["inputs"][:, -1], batch["forcings_in"][:, -1], batch["static"]
        z = model.encode(x, f, st)
        out["latent"].append(latent_invariants(z, patches).float().numpy())
        f_next = batch["forcings_tgt"][:, 0]
        act = []
        if tisr is not None:
            act.append(patches.pool(model.graphs.grid_to_latent_mean(f_next[:, tisr:tisr + 1]))[..., 0])
        if season:
            act.append(f_next[:, season].mean((-2, -1)))
        out["action"].append(torch.cat(act, -1).numpy() if act else np.zeros((len(items), 0), np.float32))
        phys = stats.unnormalize(x.numpy(), dyn)  # [B, C, H, W]
        out["fields"].append(phys[:, field_idx].astype(np.float32))
        e = coarsen(phys[:, e_idx], 2) * e_w[None, :, None, None] * area_c[None, None, :, None]
        out["energy"].append(e.reshape(len(items), -1).astype(np.float32))
        out["times"].append(batch["init_time"].numpy())
        if log and (start // batch_size) % 100 == 0:
            log(f"  encoded {min(start + batch_size, n)}/{n}")
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["times"] = res["times"].astype("datetime64[s]")
    res["patch_centres"] = patches.centres.cpu().numpy()
    res["lat"], res["lon"] = np.asarray(dataset.lat), np.asarray(dataset.lon)
    return res


@torch.no_grad()
def world_model_errors(model, dataset, stats, ks: Sequence[int], steps: int, key: str = "geopotential@500",
                       sample: bool = False) -> np.ndarray:
    """Lat-weighted RMSE of `key` at lead `steps` for windows ks (deterministic rollouts)."""
    from .causes import latent_rollout
    layout = model.layout
    c = [ch.key for ch in layout.targets].index(key)
    w = torch.as_tensor(normalized_latitude_weights(dataset.lat), dtype=torch.float32)
    errs = []
    for k in ks:
        batch = torch.utils.data.default_collate([dataset[k]])
        z = latent_rollout(model, batch, steps, sample=sample)[-1]
        pred = stats.unnormalize(model.decode(z), layout.targets)[:, c]
        tgt = stats.unnormalize(batch["targets"][:, steps - 1], layout.targets)[:, c]
        errs.append(float(((pred - tgt) ** 2 * w[:, None]).mean().sqrt()))
    return np.asarray(errs)


# -------------------------------------------------------------- seasonality
def harmonics(times: np.ndarray) -> np.ndarray:
    """[N, 11]: constant, three annual and two diurnal harmonics."""
    t = np.asarray(times, "datetime64[s]").astype(np.int64) / 86400.0
    yr = 2 * np.pi * t / 365.2425
    day = 2 * np.pi * t
    cols = [np.ones_like(t)]
    for k in (1, 2, 3):
        cols += [np.sin(k * yr), np.cos(k * yr)]
    for k in (1, 2):
        cols += [np.sin(k * day), np.cos(k * day)]
    return np.stack(cols, 1)


def deseasonalise(x: np.ndarray, times: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Anomaly from a harmonic climatology fitted on the rows `fit` (bool [N])."""
    flat = x.reshape(len(x), -1).astype(np.float64)
    h = harmonics(times)
    coef, *_ = np.linalg.lstsq(h[fit], flat[fit], rcond=None)
    return (flat - h @ coef).reshape(x.shape).astype(np.float32)


def standardise(x: np.ndarray, fit: np.ndarray) -> np.ndarray:
    flat = x.reshape(len(x), -1)
    mu, sd = flat[fit].mean(0), flat[fit].std(0) + 1e-6
    return ((flat - mu) / sd).reshape(x.shape).astype(np.float32)


def pca(x: np.ndarray, fit: np.ndarray, k: int):
    """Projection of all rows on the top-k principal directions of rows `fit`;
    returns (scores [N, k], variances [k])."""
    flat = torch.as_tensor(x.reshape(len(x), -1), dtype=torch.float32)
    mu = flat[torch.as_tensor(fit)].mean(0)
    k = min(k, int(fit.sum()) - 1, flat.shape[1])
    _, s, v = torch.svd_lowrank(flat[torch.as_tensor(fit)] - mu, q=min(k + 10, flat.shape[1]), niter=4)
    scores = (flat - mu) @ v[:, :k]
    return scores.numpy(), (s[:k] ** 2 / max(int(fit.sum()) - 1, 1)).numpy()


# ------------------------------------------------------- clusters, MSM
def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 60):
    """Lloyd's k-means with k-means++ initialisation; returns (labels, centres)."""
    gen = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    centres = [x[gen.integers(len(x))]]
    d2 = ((x - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        centres.append(x[gen.choice(len(x), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((x - centres[-1]) ** 2).sum(1))
    c = np.stack(centres)
    for _ in range(iters):
        lab = (-2 * x @ c.T + (c ** 2).sum(1)).argmin(1)
        new = np.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return (-2 * x @ c.T + (c ** 2).sum(1)).argmin(1), c


def transition_counts(labels: np.ndarray, pairs: np.ndarray, n: int) -> np.ndarray:
    counts = np.zeros((n, n))
    np.add.at(counts, (labels[pairs[:, 0]], labels[pairs[:, 1]]), 1.0)
    return counts


def commute_times(counts: np.ndarray):
    """Commute times H_ij + H_ji of the Markov chain on the largest strongly
    connected set of clusters; returns (C [n, n], nan outside; keep [n] bool)."""
    n = len(counts)
    _, comp = connected_components(counts > 0, directed=True, connection="strong")
    keep = comp == np.bincount(comp).argmax()
    sub = counts[np.ix_(keep, keep)]
    p = sub / sub.sum(1, keepdims=True)
    m = len(p)
    vals, vecs = np.linalg.eig(p.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    pi = np.abs(pi) / np.abs(pi).sum()
    z = np.linalg.inv(np.eye(m) - p + np.outer(np.ones(m), pi))
    h = (np.diag(z)[None, :] - z) / pi[None, :]
    out = np.full((n, n), np.nan)
    out[np.ix_(keep, keep)] = h + h.T
    return out, keep


def laplacian_spectrum(counts: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Non-zero eigenvalues (ascending) of the normalised Laplacian of W = N + N^T."""
    w = counts[np.ix_(keep, keep)]
    w = w + w.T
    d = w.sum(1)
    lap = np.eye(len(w)) - w / np.sqrt(np.outer(d, d))
    return np.sort(np.linalg.eigvalsh(lap))[1:]


# -------------------------------------------------------------------- tests
def _boot_ids(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def _ci(samples: np.ndarray, point: float) -> Dict[str, float]:
    lo, hi = np.nanpercentile(samples, [5, 95])
    return {"mean": float(point), "lo": float(lo), "hi": float(hi)}


def centroid_spearman(emb: np.ndarray, labels: np.ndarray, ct: np.ndarray, n_boot: int = 200, seed: int = 0):
    """T1: Spearman(centroid distance, commute time) over cluster pairs, and its
    cluster-bootstrap samples (the same resamples for every representation, so
    differences are paired)."""
    ids = np.array([c for c in np.unique(labels) if np.isfinite(ct[c]).any()])
    cent = np.stack([emb[labels == c].mean(0) for c in ids])
    dist = np.sqrt(((cent[:, None] - cent[None]) ** 2).sum(-1))
    cts = ct[np.ix_(ids, ids)]

    def rho(sel):
        a, b = np.triu_indices(len(sel), 1)
        a, b = sel[a], sel[b]
        ok = (a != b) & np.isfinite(cts[a, b])
        return spearmanr(dist[a[ok], b[ok]], cts[a[ok], b[ok]]).statistic if ok.sum() > 3 else np.nan

    point = rho(np.arange(len(ids)))
    samples = np.array([rho(row) for row in _boot_ids(len(ids), n_boot, seed)])
    return point, samples


def lag_profile(emb: np.ndarray, times: np.ndarray, lags: Sequence[int], seed: int = 0) -> Dict[str, object]:
    """Mean distance at each lag (steps) over a contiguous year, relative to the
    mean distance of random pairs, and Spearman of pair distance against lag."""
    t = np.asarray(times, "datetime64[s]").astype(np.int64) // 3600
    pos = {int(v): i for i, v in enumerate(t)}
    gen = np.random.default_rng(seed)
    rand = gen.integers(0, len(emb), (4000, 2))
    scale = np.sqrt(((emb[rand[:, 0]] - emb[rand[:, 1]]) ** 2).sum(-1)).mean()
    means, all_d, all_l = [], [], []
    for lag in lags:
        i = np.array([k for k, v in enumerate(t) if int(v) + STEP_H * lag in pos])
        j = np.array([pos[int(t[k]) + STEP_H * lag] for k in i])
        d = np.sqrt(((emb[i] - emb[j]) ** 2).sum(-1))
        means.append(float(d.mean() / scale))
        sub = gen.choice(len(d), min(len(d), 300), replace=False)
        all_d.append(d[sub])
        all_l.append(np.full(len(sub), lag))
    rho = spearmanr(np.concatenate(all_d), np.concatenate(all_l)).statistic
    return {"lags": list(map(int, lags)), "relative_distance": means, "spearman_lag": float(rho)}


def spectrum_slope(emb: np.ndarray, lam: np.ndarray, m: int) -> Dict[str, object]:
    """T2 geometry: slope of log variance of the top-m principal directions against log lambda_k."""
    x = emb - emb.mean(0)
    var = np.sort(np.linalg.eigvalsh(x.T @ x / (len(x) - 1)))[::-1]
    m = min(m, len(var), len(lam))
    var, lam_m = np.clip(var[:m], 1e-12, None), np.clip(lam[:m], 1e-12, None)
    slope = np.polyfit(np.log(lam_m), np.log(var), 1)[0]
    return {"slope": float(slope), "variance": var.tolist(), "lambda": lam_m.tolist()}


def h0_persistence(emb: np.ndarray, seed: int = 0, top: int = 10) -> Dict[str, object]:
    """T2 topology: H0 deaths (single-linkage merge heights) relative to a
    Gaussian cloud with the same mean and covariance; `excess` counts deaths
    above the null's largest, i.e. components that a single blob does not make."""
    gen = np.random.default_rng(seed)
    x = np.asarray(emb, np.float64)
    deaths = np.sort(linkage(x, "single")[:, 2])[::-1]
    null = gen.multivariate_normal(x.mean(0), np.cov(x.T) + 1e-9 * np.eye(x.shape[1]), len(x))
    null_deaths = np.sort(linkage(null, "single")[:, 2])[::-1]
    scale = np.median(deaths)
    return {"top": (deaths[:top] / scale).tolist(), "null_top": (null_deaths[:top] / scale).tolist(),
            "excess": int((deaths > null_deaths[0]).sum())}


def h1_persistence(emb: np.ndarray, max_points: int = 600, seed: int = 0) -> Optional[Dict[str, object]]:
    """Longest H1 bars (ripser, if installed), relative to the median H0 death."""
    try:
        from ripser import ripser
    except ImportError:
        return None
    gen = np.random.default_rng(seed)
    x = emb[gen.choice(len(emb), min(max_points, len(emb)), replace=False)]
    dg = ripser(x, maxdim=1)["dgms"]
    h0 = dg[0][np.isfinite(dg[0][:, 1]), 1]
    life = np.sort(dg[1][:, 1] - dg[1][:, 0])[::-1]
    return {"top": (life[:5] / np.median(h0)).tolist(), "count": int(len(life))}


def residence_time(emb: np.ndarray, k: int = 8, seed: int = 0) -> Dict[str, float]:
    """Mean run length (steps) in k-means regimes of a contiguous sequence, and
    the self-transition fraction."""
    lab, _ = kmeans(emb, k, seed)
    change = np.flatnonzero(np.diff(lab) != 0)
    runs = np.diff(np.concatenate([[-1], change, [len(lab) - 1]]))
    return {"mean_run": float(runs.mean()), "self_fraction": float((np.diff(lab) == 0).mean())}


def analog_errors(lib: np.ndarray, test: np.ndarray, lib_future: np.ndarray, test_future: np.ndarray,
                  lat_w: np.ndarray, k: int = 10) -> np.ndarray:
    """RMSE [N_test, V] of the mean of the k nearest library states' successors."""
    lt, tt = torch.as_tensor(lib, dtype=torch.float32), torch.as_tensor(test, dtype=torch.float32)
    nbr = torch.cat([torch.cdist(tt[i:i + 256], lt).topk(k, largest=False).indices
                     for i in range(0, len(tt), 256)]).numpy()
    fc = lib_future[nbr].mean(1)  # [N_test, V, H, W]
    return np.sqrt(((fc - test_future) ** 2 * lat_w[None, None, :, None]).mean((-2, -1)))


def block_bootstrap_change(err: np.ndarray, ref: np.ndarray, n_boot: int, block: int, seed: int):
    """mean(err)/mean(ref) - 1 per column with a moving-block bootstrap over rows."""
    n = len(err)
    gen = np.random.default_rng(seed)
    starts = gen.integers(0, max(n - block, 1), (n_boot, int(np.ceil(n / block))))
    idx = (starts[..., None] + np.arange(block)).reshape(n_boot, -1)[:, :n] % n
    point = err.mean(0) / ref.mean(0) - 1
    samples = err[idx].mean(1) / ref[idx].mean(1) - 1
    lo, hi = np.percentile(samples, [5, 95], axis=0)
    return {"mean": point.tolist(), "lo": lo.tolist(), "hi": hi.tolist()}


def blocking_and_jet(z500: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                     sector=(-30.0, 40.0)) -> Dict[str, np.ndarray]:
    """Atlantic-European regimes from z500 [N, H, W] (m^2/s^2):
    blocked = Tibaldi-Molteni reversal (GHGS > 0, GHGN < -10 m/deg) at some
    sector longitude; jet = latitude of the strongest poleward z500 drop
    (geostrophic jet), 30-70N, sector mean."""
    z = z500 / 9.80665
    lon180 = (np.asarray(lon) + 180) % 360 - 180
    sec = (lon180 >= sector[0]) & (lon180 <= sector[1])
    row = lambda phi: int(np.argmin(np.abs(np.asarray(lat) - phi)))
    s, o, nn = row(40), row(60), row(80)
    ghgs = (z[:, o] - z[:, s]) / (lat[o] - lat[s])
    ghgn = (z[:, nn] - z[:, o]) / (lat[nn] - lat[o])
    blocked = ((ghgs > 0) & (ghgn < -10))[:, sec].any(1)
    band = np.flatnonzero((np.asarray(lat) >= 30) & (np.asarray(lat) <= 70))
    zs = z[:, band][:, :, sec].mean(-1)
    grad = -np.diff(zs, axis=1) * np.sign(lat[band[1]] - lat[band[0]])
    mid = 0.5 * (np.asarray(lat)[band[1:]] + np.asarray(lat)[band[:-1]])
    return {"blocked": blocked, "jet_lat": mid[grad.argmax(1)]}


# ------------------------------------------------------------------- driver
REGIONS = {"north_atlantic": (55.0, -30.0), "north_pacific": (50.0, -170.0)}


def nearest_patch(centres: np.ndarray, lat: float, lon: float) -> int:
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    p = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return int(np.argmax(centres @ p))


def transition_pairs(times: np.ndarray, rows: np.ndarray, lag: int = 1) -> np.ndarray:
    """[M, 2] row pairs (i, j), both in `rows` (bool), with t_j = t_i + lag steps."""
    t = np.asarray(times, "datetime64[s]").astype(np.int64)
    pos = {int(v): i for i, v in enumerate(t)}
    out = [(i, pos[int(t[i]) + 3600 * STEP_H * lag]) for i in np.flatnonzero(rows)
           if int(t[i]) + 3600 * STEP_H * lag in pos]
    out = [(i, j) for i, j in out if rows[j]]
    return np.asarray(out, np.int64).reshape(-1, 2)


def _top_pcs(x: np.ndarray, m: int) -> np.ndarray:
    x = x - x.mean(0)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:m].T


def analyse(store: Dict[str, np.ndarray], arms: Dict[str, "CommuteConfig"], train_end: str = "2018-01-01",
            test_end: str = "2019-01-01", n_clusters: int = 64, leads: Sequence[int] = (4, 12, 20),
            pca_dim: int = 128, topo_dim: int = 16, analogs: int = 10, n_boot: int = 200, seed: int = 0,
            log=print) -> Dict[str, object]:
    """Builds every representation and runs T1-T3 (see the module docstring)."""
    from ..models.commute_time import embed_all, fit_head
    times = np.asarray(store["times"], "datetime64[s]")
    train = times < np.datetime64(train_end)
    test = (~train) & (times < np.datetime64(test_end))
    n = len(times)
    lat_anom = deseasonalise(store["latent"], times, train)                  # [N, S, P, c]
    raw = standardise(lat_anom, train).reshape(n, -1)
    raw_pcs, _ = pca(raw, train, pca_dim)
    head_global = raw_pcs / raw_pcs[train, 0].std()
    s, p, c = lat_anom.shape[1:]
    head_patch = standardise(lat_anom.transpose(0, 2, 1, 3).reshape(n, p, s * c), train)
    energy, _ = pca(deseasonalise(store["energy"], times, train), train, pca_dim)
    action = standardise(store["action"], train) if store["action"].shape[1] else None
    reps: Dict[str, np.ndarray] = {"raw_z": raw, "energy": energy}
    patch_reps: Dict[str, np.ndarray] = {"raw_z": head_patch}
    histories = {}
    train_pairs = transition_pairs(times, train)
    for name, cfg in arms.items():
        log(f"head {name}: {cfg}")
        x = head_patch if cfg.mode == "patch" else head_global
        head, histories[name] = fit_head(x, train_pairs, action, cfg, log=log)
        emb = embed_all(head, x)
        reps[name] = emb.reshape(n, -1)
        if emb.ndim == 3:
            patch_reps[name] = emb

    # Markov state model on all years' transitions; two partitions
    all_pairs = transition_pairs(times, np.ones(n, bool))
    partitions = {"latent": _top_pcs(raw_pcs, 32), "energy": _top_pcs(energy, 32)}
    msm = {}
    for pname, feats in partitions.items():
        labels, _ = kmeans(feats, n_clusters, seed)
        counts = transition_counts(labels, all_pairs, n_clusters)
        ct, keep = commute_times(counts)
        msm[pname] = {"labels": labels, "ct": ct, "keep": keep, "lambda": laplacian_spectrum(counts, keep)}
    lam = msm["latent"]["lambda"]

    # analog library: rows with every lead inside the store
    fields, lat = store["fields"], np.asarray(store["lat"])
    lat_w = normalized_latitude_weights(lat).astype(np.float32)
    t_int = times.astype(np.int64)
    pos = {int(v): i for i, v in enumerate(t_int)}
    succ = {L: np.array([pos.get(int(v) + 3600 * STEP_H * L, -1) for v in t_int]) for L in leads}
    ok = np.all([succ[L] >= 0 for L in leads], 0)
    lib, tst = np.flatnonzero(train & ok), np.flatnonzero(test & ok)
    regimes = blocking_and_jet(fields[tst, 0], lat, np.asarray(store["lon"]))
    jet_lo, jet_hi = np.percentile(regimes["jet_lat"], [100 / 3, 200 / 3])
    subsets = {"all": np.ones(len(tst), bool), "blocked": regimes["blocked"], "not_blocked": ~regimes["blocked"],
               "jet_south": regimes["jet_lat"] <= jet_lo, "jet_north": regimes["jet_lat"] >= jet_hi}

    test_rows = np.flatnonzero(test)
    out: Dict[str, object] = {
        "rows": {"train": int(train.sum()), "test": int(test.sum()), "train_pairs": int(len(train_pairs)),
                 "analog_library": int(len(lib)), "analog_tests": int(len(tst))},
        "msm": {k: {"clusters": n_clusters, "kept": int(v["keep"].sum())} for k, v in msm.items()},
        "leads_h": [STEP_H * L for L in leads], "variables": list(ANALOG_KEYS),
        "regime_counts": {k: int(v.sum()) for k, v in subsets.items()},
        "arms": {}, "regional": {},
    }
    t1_samples, t3_err = {}, {}
    for name, emb in reps.items():
        log(f"tests {name} (dim {emb.shape[1]})")
        r: Dict[str, object] = {"dim": int(emb.shape[1])}
        t1 = {}
        for pname, m in msm.items():
            point, samples = centroid_spearman(emb[test], m["labels"][test], m["ct"], n_boot, seed)
            t1[pname] = _ci(samples, point)
            t1_samples[(name, pname)] = samples
        r["t1"] = t1
        r["lag"] = lag_profile(emb[test_rows], times[test_rows], (1, 2, 4, 8, 12, 20, 28, 40), seed)
        low = _top_pcs(emb[test_rows], min(topo_dim, emb.shape[1]))
        r["t2"] = {"spectrum": spectrum_slope(emb[test], lam, topo_dim), "h0": h0_persistence(low, seed),
                   "h1": h1_persistence(low, seed=seed), "residence": residence_time(low, 8, seed)}
        errs = np.stack([analog_errors(emb[lib], emb[tst], fields[succ[L][lib]], fields[succ[L][tst]], lat_w,
                                       analogs) for L in leads], 1)                  # [N_test, L, V]
        t3_err[name] = errs
        r["t3"] = {"rmse": errs.mean(0).tolist()}
        if "wm_err" in store and len(store["wm_err"]):
            rows = np.asarray(store["wm_rows"])
            nxt = succ[1][rows] if 1 in succ else np.array([pos.get(int(t_int[i]) + 3600 * STEP_H, -1) for i in rows])
            good = nxt >= 0
            speed = np.sqrt(((emb[nxt[good]] - emb[rows[good]]) ** 2).sum(-1))
            d = torch.cdist(torch.as_tensor(emb[rows], dtype=torch.float32), torch.as_tensor(emb[lib], dtype=torch.float32))
            rarity = d.topk(analogs, largest=False).values.mean(1).numpy()
            r["predictability"] = {
                "speed_vs_error": float(spearmanr(speed, store["wm_err"][good]).statistic),
                "rarity_vs_error": float(spearmanr(rarity, store["wm_err"]).statistic), "inits": int(len(rows))}
        out["arms"][name] = r
    for name, r in out["arms"].items():
        for ref in ("raw_z", "energy"):
            if name == ref:
                continue
            for pname in msm:
                diff = t1_samples[(name, pname)] - t1_samples[(ref, pname)]
                r["t1"][f"{pname}_minus_{ref}"] = _ci(diff, r["t1"][pname]["mean"] - out["arms"][ref]["t1"][pname]["mean"])
            r["t3"][f"vs_{ref}"] = {
                sub: block_bootstrap_change(t3_err[name][mask].reshape(int(mask.sum()), -1),
                                            t3_err[ref][mask].reshape(int(mask.sum()), -1), n_boot, 28, seed)
                for sub, mask in subsets.items() if mask.sum() > 10}
        r["history"] = histories.get(name, [])
    for region, (la, lo) in REGIONS.items():
        k = nearest_patch(np.asarray(store["patch_centres"]), la, lo)
        reg = {"patch": k}
        for name, emb in patch_reps.items():
            low = _top_pcs(emb[test_rows, k].reshape(len(test_rows), -1), topo_dim)
            reg[name] = {"h0": h0_persistence(low, seed), "residence": residence_time(low, 8, seed)}
        out["regional"][region] = reg
    return out
