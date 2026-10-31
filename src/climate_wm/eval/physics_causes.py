"""Stage 9 cause analysis: does the physics world model rest on the coupled
processes of a climate model, and do they act the way the coupling says?

A climate model couples the thermodynamic equation, the equations of motion,
radiative transfer, the conservation of water mass and the surface heat
budget. The arrows between them are:
  - advection (motion -> thermodynamics, motion -> water);
  - density (thermodynamics -> motion);
  - latent heating (water -> thermodynamics);
  - moisture (water -> radiation);
  - heating and cooling (radiation -> thermodynamics);
  - heat energy (surface <-> radiation);
  - precipitation and evaporation (surface <-> water).
In the equations (pressure coordinates here, sigma in the usual figure), the
blocks exchange T' (thermodynamics -> momentum, through the pressure
gradient), u and omega (momentum -> thermodynamics and moisture), T and T_v
(-> hydrostatic balance) and Phi, Phi_b (hydrostatic balance -> momentum);
the learned column physics adds Psi_theta to every block. Each arrow is cut
at evaluation time, without retraining, by one intervention (INTERVENTIONS;
`edge` names the arrow; the core arrows are primitive_core.CUTS and TERMS). The cuts from water and from the sun
to the heating are path-specific: only the heating output of the column
network sees the held input, and its other outputs see the state. In N1,
evaporation only diagnoses precipitation, so the surface -> water arrow has no
path in the model (MISSING).

The graph has loops, so single cuts are not the whole story:
  pairs (PAIRS): the non-additivity of two cuts in state space,
      eta = |D_AB - D_A - D_B| / (|D_A| + |D_B|),   D_X = forecast_X - forecast,
    which is zero for independent paths, and the share of A's effect left once
    B is cut,
      rho(A | B) = |forecast_AB - forecast_B| / |D_A|   (B mediates A: rho -> 0);
  propagation: the response |D_X| per block of the equations (GROUPS:
    thermodynamics, momentum, hydrostatic balance, water) at every lead, so
    the order in which an intervention reaches the blocks can be read against
    the arrows;
  invariance: the same tests on two years (scripts/report_physics_causes.py).
Skill: RMSE of the headline variables and the channel-mean normalised MSE per
lead, paired with the intact model (bootstrap 95%). A blow-up (non-finite, or
the normalised MSE above `cap`) is counted, and the run is dropped from the
pairs from that lead. Every intervention also reports the 2 m temperature
response on day and night points (TOA sunlight at the valid time) and its
correlation with the solar energy received since the start; for do(TISR = 0)
that energy is the energy withheld.
"""

from __future__ import annotations

import copy
import dataclasses
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..data import forcings as forcing_lib
from ..models.physics_world_model import PhysicsModelConfig, PhysicsWorldModel
from ..models.primitive_core import TERMS
from .graphcast_metrics import HEADLINE_KEYS, normalized_latitude_weights, weighted_mean

TISR = "toa_incident_solar_radiation"
PAIR_STEPS = (1, 4, 12)  # 6, 24, 72 h: the leads of the pair tests


@dataclasses.dataclass(frozen=True)
class Intervention:
    edge: str                                  # the arrow of the coupling diagram it cuts
    off_terms: Tuple[str, ...] = ()            # core terms (primitive_core.TERMS)
    cuts: Tuple[str, ...] = ()                 # core coupling paths (primitive_core.CUTS)
    drag_off: bool = False
    carry_off: bool = False
    knock: Tuple[str, ...] = ()                # column outputs set to zero (COLUMN_KNOCKS)
    paths: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()  # (column output group, held inputs)
    sun: bool = False                          # do(TISR = 0) for every consumer
    density_scale: Optional[float] = None      # soft intervention on T' -> momentum

    def __add__(self, other: "Intervention") -> "Intervention":
        paths: Dict[str, Tuple[str, ...]] = {}
        for group, inputs in self.paths + other.paths:
            paths[group] = tuple(dict.fromkeys(paths.get(group, ()) + inputs))
        uni = lambda a, b: tuple(dict.fromkeys(a + b))
        return Intervention(f"{self.edge} + {other.edge}", uni(self.off_terms, other.off_terms),
                            uni(self.cuts, other.cuts), self.drag_off or other.drag_off,
                            self.carry_off or other.carry_off, uni(self.knock, other.knock),
                            tuple(paths.items()), self.sun or other.sun,
                            min((x for x in (self.density_scale, other.density_scale) if x is not None),
                                default=None))

    def needs_column(self) -> bool:
        return bool(self.knock or self.paths or self.sun)


I = Intervention
INTERVENTIONS: Dict[str, Intervention] = {
    "no_temp_advection": I("motion -> thermodynamics (u, omega advect T)", cuts=("temp_advection",)),
    "no_adiabatic": I("motion -> thermodynamics (kappa T omega / p)", off_terms=("adiabatic",)),
    "no_density": I("thermodynamics -> motion (T' in the pressure gradient: density)", cuts=("density",)),
    "weak_density": I("thermodynamics -> motion, T' coupling 10% weaker (soft: the full cut blows up)",
                      density_scale=0.9),
    "no_virtual": I("water -> hydrostatic balance (T_v: buoyancy of water vapour)", cuts=("virtual",)),
    "no_moisture": I("motion -> water (u, omega advect q)", off_terms=("moisture",)),
    "no_pgf": I("hydrostatic balance -> motion (the whole pressure gradient: Phi, Phi_b)", off_terms=("pgf",)),
    "no_advection": I("motion -> motion (horizontal advection of momentum, T and Phi_b)",
                      off_terms=("advection",)),
    "no_coriolis": I("rotation -> motion (Coriolis)", off_terms=("coriolis",)),
    "no_vertical": I("motion -> all (vertical advection)", off_terms=("vertical",)),
    "no_drag": I("surface -> motion (boundary-layer friction)", drag_off=True),
    "no_carry": I("initial residual (unresolved scales, offsets)", carry_off=True),
    "no_heating": I("radiation + latent heating -> thermodynamics (column Q)", knock=("heating",)),
    "no_moistening": I("water sources and sinks (column S)", knock=("moistening",)),
    "no_momentum": I("column momentum forcing -> motion", knock=("momentum",)),
    "no_surface_pressure": I("column -> surface pressure (P_b)", knock=("surface_pressure",)),
    "q_to_heating": I("water -> thermodynamics (latent heating) and water -> radiation (moisture)",
                      paths=(("heating", ("q",)),)),
    "t_to_moistening": I("thermodynamics -> water (saturation, condensation)",
                         paths=(("moistening", ("temp",)),)),
    "sun_off": I("sun -> radiation: do(TISR = 0)", sun=True),
    "sun_to_heating": I("radiation -> thermodynamics (heating and cooling), the sun's path only",
                        paths=(("heating", ("sun",)),)),
    "ocean_column": I("surface (land / ocean) -> radiation and column physics", paths=(("all", ("land",)),)),
}
PAIRS: Tuple[Tuple[str, str], ...] = (
    ("no_temp_advection", "weak_density"),  # thermodynamics <-> motion (u, omega and T')
    ("no_moisture", "no_virtual"),       # motion -> water -> hydrostatic balance -> motion (T_v)
    ("no_moisture", "q_to_heating"),     # motion -> water -> thermodynamics (the latent-heat loop)
    ("t_to_moistening", "q_to_heating"),  # thermodynamics <-> water (condensation)
    ("sun_off", "no_heating"),           # does the column heating carry the sun? (mediation)
    ("sun_off", "q_to_heating"),         # water -> radiation: moisture modulates the solar heating
    ("sun_off", "ocean_column"),         # surface <-> radiation
)
MISSING = {"surface <-> water (precipitation, evaporation)":
           "evaporation E only diagnoses precipitation (P = E - int S dp / g); no tendency reads E, "
           "and nothing stores surface water or heat"}


GROUPS = ("thermodynamics", "momentum", "hydrostatic", "water")


def groups(layout) -> Dict[str, List[int]]:
    """Target channels per block of the equations (GROUPS): T; winds and omega;
    geopotential and sea-level pressure (hydrostatic balance and mass); q and
    precipitation."""
    out: Dict[str, List[int]] = {g: [] for g in GROUPS}
    for j, c in enumerate(layout.targets):
        k = c.key
        if k.startswith("temperature@") or k == "2m_temperature":
            out["thermodynamics"].append(j)
        elif k.startswith("specific_humidity@") or k == "total_precipitation_6hr":
            out["water"].append(j)
        elif k.startswith("geopotential@") or k == "mean_sea_level_pressure":
            out["hydrostatic"].append(j)
        else:  # u, v, vertical velocity, 10 m winds
            out["momentum"].append(j)
    return out


def available(name: str, model: PhysicsWorldModel) -> bool:
    """Whether the intervention (or pair "A+B") has something to cut in `model`."""
    if "+" in name:
        return all(available(n, model) for n in name.split("+"))
    iv = INTERVENTIONS[name]
    if iv.needs_column() and model.column is None:
        return False
    if (iv.sun or any("sun" in inputs for _, inputs in iv.paths)) and not model.cfg.column.use_forcings:
        return False
    if iv.drag_off and model.cfg.core.surface_drag_days <= 0:
        return False
    if iv.carry_off and not model.cfg.carry_residual:
        return False
    return all(t in model.cfg.core.terms for t in iv.off_terms)


def intervention(name: str) -> Intervention:
    if "+" in name:
        a, b = name.split("+")
        return INTERVENTIONS[a] + INTERVENTIONS[b]
    return INTERVENTIONS[name]


def build(model_cfg: dict, state: dict, layout, lat, lon, iv: Optional[Intervention], device="cpu"):
    """The model of a checkpoint (model config dict + state dict) under an intervention."""
    cfg = copy.deepcopy(model_cfg)
    cfg.pop("type", None)
    if iv is not None:
        core = cfg.setdefault("core", {})
        if iv.off_terms:
            core["terms"] = [t for t in core.get("terms", TERMS) if t not in iv.off_terms]
        if iv.cuts:
            core["cuts"] = list(dict.fromkeys(list(core.get("cuts", ())) + list(iv.cuts)))
        if iv.drag_off:
            core["surface_drag_days"] = 0.0
        if iv.density_scale is not None:
            core["density_scale"] = iv.density_scale
        if iv.carry_off:
            cfg["carry_residual"] = False
    model = PhysicsWorldModel(PhysicsModelConfig.from_dict(cfg), layout, lat, lon)
    missing, _ = model.load_state_dict(state, strict=False)
    params = {n for n, _ in model.named_parameters()}
    if params & set(missing):
        raise KeyError(f"checkpoint lacks parameters {sorted(params & set(missing))}")
    if iv is not None:
        model.column_knock = set(iv.knock)
        model.column_paths = dict(iv.paths)
        if iv.sun:
            model.forcing_override = {TISR: 0.0}
    return model.to(device).eval()


@torch.no_grad()
def rollout(model: PhysicsWorldModel, batch: Dict[str, torch.Tensor], steps: int, sun_off: bool = False):
    """Normalised forecast [B, L, C, H, W]; with `sun_off` the surface head's forcings
    also see TISR = 0 (the column's come from model.forcing_override)."""
    if sun_off:
        batch = dict(batch)
        j = model.forcing_keys.index(TISR)
        f = batch["forcings_tgt"].clone()
        f[:, :, j] = -model.forcing_mean[j] / model.forcing_std[j]
        batch["forcings_tgt"] = f
    return model.forward_rollout(batch, steps).float()


def _wcorr(x: torch.Tensor, y: torch.Tensor, lat_w: torch.Tensor) -> torch.Tensor:
    """Area-weighted Pearson correlation over (lat, lon) of [..., H, W] fields."""
    xa = x - weighted_mean(x, lat_w)[..., None, None]
    ya = y - weighted_mean(y, lat_w)[..., None, None]
    den = (weighted_mean(xa.square(), lat_w) * weighted_mean(ya.square(), lat_w)).sqrt()
    return weighted_mean(xa * ya, lat_w) / den.clamp_min(1e-12)


def _norm(d: torch.Tensor, chans: Sequence[int], lat_w: torch.Tensor) -> torch.Tensor:
    """RMS over the channels `chans` and area of d [B, C, H, W] -> [B]."""
    return weighted_mean(d[:, list(chans)].square(), lat_w).mean(-1).sqrt()


class _Scorer:
    def __init__(self, layout, stats, dataset, batch, steps: int, cap: float):
        self.keys = [c.key for c in layout.targets]
        self.head = [k for k in HEADLINE_KEYS if k in self.keys]
        self.kidx = [self.keys.index(k) for k in self.head]
        self.std = torch.tensor([float(stats.std[k]) for k in self.head])
        self.t2m = self.keys.index("2m_temperature") if "2m_temperature" in self.keys else None
        self.t2m_std = float(stats.std["2m_temperature"]) if self.t2m is not None else 1.0
        self.groups = groups(layout)
        self.lat_w = torch.tensor(normalized_latitude_weights(dataset.lat), dtype=torch.float32)
        self.cap, self.steps = cap, steps
        self.truth = batch["targets"][:, :steps].float().cpu()
        init = batch["init_time"].cpu().numpy().astype(np.int64)
        lead_s = 6 * 3600 * np.arange(1, steps + 1)
        sun = np.stack([forcing_lib.toa_incident_solar_radiation(
            (t + lead_s).astype("datetime64[s]"), dataset.lat, dataset.lon) for t in init])  # [B, L, H, W]
        self.day = torch.from_numpy(sun > 0).float()
        self.energy = torch.from_numpy(np.cumsum(sun, 1)).float()  # J/m^2, 1 h at every 6 h sample

    def skill(self, pred: torch.Tensor) -> Dict[str, object]:
        d2 = (pred - self.truth).square()
        mse = weighted_mean(d2, self.lat_w)                        # [B, L, C]
        all_mse = mse.mean(-1)                                     # [B, L]
        blown = (~torch.isfinite(all_mse)) | (all_mse > self.cap)
        blown = torch.cummax(blown.int(), 1).values.bool()
        rmse = mse[:, :, self.kidx].sqrt() * self.std              # [B, L, K]
        nan = torch.tensor(float("nan"))
        all_mse = torch.where(blown, nan, all_mse)
        rmse = torch.where(blown[..., None], nan, rmse)
        first = [int(6 * (np.argmax(b) + 1)) if b.any() else None for b in blown.numpy()]
        return {"all": all_mse.tolist(), "rmse": {k: rmse[:, :, j].tolist() for j, k in enumerate(self.head)},
                "blown": blown.tolist(), "first_blowup_h": first}

    def response(self, pred: torch.Tensor, full: torch.Tensor, blown: torch.Tensor) -> Dict[str, object]:
        d = pred - full
        nan = torch.tensor(float("nan"))
        out = {}
        for g, chans in self.groups.items():
            r = torch.stack([_norm(d[:, l], chans, self.lat_w) for l in range(self.steps)], 1)  # [B, L]
            out[g] = torch.where(blown, nan, r).tolist()
        if self.t2m is not None:
            dt = d[:, :, self.t2m] * self.t2m_std  # K, [B, L, H, W]
            w = self.lat_w[:, None]
            mean_on = lambda m: (dt * m * w).sum((-2, -1)) / (m * w).sum((-2, -1)).clamp_min(1e-12)
            out["t2m_day_K"] = torch.where(blown, nan, mean_on(self.day)).tolist()
            out["t2m_night_K"] = torch.where(blown, nan, mean_on(1 - self.day)).tolist()
            out["t2m_corr_energy"] = torch.where(blown, nan, _wcorr(dt, self.energy, self.lat_w)).tolist()
        return out


def _pair_steps(steps: int) -> List[int]:
    return [s for s in PAIR_STEPS if s <= steps] or [steps]


def _pair(fa, fb, fab, full, sc: _Scorer) -> Dict[str, object]:
    """eta and rho per group at PAIR_STEPS (module docstring); inputs [B, L, C, H, W]."""
    out: Dict[str, object] = {"eta": {}, "rho_a_given_b": {}, "rho_b_given_a": {}}
    for g, chans in sc.groups.items():
        eta, ra, rb = [], [], []
        for s in _pair_steps(sc.steps):
            l = s - 1
            da, db, dab = fa[:, l] - full[:, l], fb[:, l] - full[:, l], fab[:, l] - full[:, l]
            na, nb = _norm(da, chans, sc.lat_w), _norm(db, chans, sc.lat_w)
            eta.append(_norm(dab - da - db, chans, sc.lat_w) / (na + nb).clamp_min(1e-12))
            ra.append(_norm(fab[:, l] - fb[:, l], chans, sc.lat_w) / na.clamp_min(1e-12))
            rb.append(_norm(fab[:, l] - fa[:, l], chans, sc.lat_w) / nb.clamp_min(1e-12))
        for key, v in (("eta", eta), ("rho_a_given_b", ra), ("rho_b_given_a", rb)):
            x = torch.stack(v, 1)
            out[key][g] = torch.where(torch.isfinite(x), x, torch.tensor(float("nan"))).tolist()
    return out


def bootstrap(rel: np.ndarray, seed: int = 0) -> Dict[str, list]:
    """Mean and 95% bootstrap interval over the first axis, ignoring NaN (blown runs)."""
    rel = np.asarray(rel, np.float64)
    ok = np.isfinite(rel)
    n = ok.sum(0)
    mean = np.where(n > 0, np.nansum(rel, 0) / np.maximum(n, 1), np.nan)
    idx = np.random.default_rng(seed).integers(0, len(rel), (2000, len(rel)))
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns: every run blew up
        boot = np.nansum(np.where(ok, rel, 0)[idx], 1) / ok[idx].sum(1)
        lo, hi = np.nanpercentile(boot, 2.5, 0), np.nanpercentile(boot, 97.5, 0)
    return {"mean": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "n": n.tolist()}


def evaluate_physics_causes(model_cfg: dict, state: dict, stats, dataset, indices: Sequence[int],
                            names: Optional[Sequence[str]] = None, steps: int = 12, cap: float = 10.0,
                            device="cpu", seed: int = 0, log=None,
                            progress_path: Optional[str] = None) -> Dict[str, object]:
    """Interventions `names` (INTERVENTIONS keys and "A+B" pairs; default: every
    available one and PAIRS) on the initialisations `indices` of `dataset`
    (`steps` 6 h targets), in one batch. With `progress_path`, every finished
    intervention is saved there and an existing file is continued (the
    forecasts a pair needs are recomputed)."""
    layout = dataset.layout
    batch = torch.utils.data.default_collate([dataset[int(i)] for i in indices])
    batch = {k: v.to(device) for k, v in batch.items()}
    sc = _Scorer(layout, stats, dataset, batch, steps, cap)
    intact = build(model_cfg, state, layout, dataset.lat, dataset.lon, None, device)
    if names is None:
        names = list(INTERVENTIONS) + [f"{a}+{b}" for a, b in PAIRS]
    names = [n for n in names if available(n, intact)]
    done: Dict[str, dict] = {}
    if progress_path is not None and os.path.exists(progress_path):
        done = torch.load(progress_path, weights_only=False)
        if log is not None:
            log(f"[causes] resumed: {len(done) - 1} of {len(names)} interventions done")
    cache: Dict[str, torch.Tensor] = {}
    full = cache["full"] = rollout(intact, batch, steps).cpu()
    if "full" not in done:
        done["full"] = {"edge": "intact model", **sc.skill(full)}
    blown_full = torch.tensor(done["full"]["blown"])
    del intact
    needed = {x for n in names if "+" in n for x in n.split("+")}

    def forecast(name):
        if name not in cache:
            iv = intervention(name)
            m = build(model_cfg, state, layout, dataset.lat, dataset.lon, iv, device)
            cache[name] = rollout(m, batch, steps, sun_off=iv.sun).cpu()
        return cache[name]

    for n, name in enumerate(names):
        if name in done:
            continue
        pred = forecast(name)
        rec = {"edge": intervention(name).edge, **sc.skill(pred)}
        blown = torch.tensor(rec["blown"]) | blown_full
        rec["response"] = sc.response(pred, full, blown)
        if "+" in name:
            a, b = name.split("+")
            rec["pair"] = _pair(forecast(a), forecast(b), pred, full, sc)
            rec["pair"]["a"], rec["pair"]["b"] = a, b
        done[name] = rec
        for key in list(cache):
            if key != "full" and key not in needed and key != name:
                del cache[key]
        if name not in needed:
            cache.pop(name, None)
        if progress_path is not None:
            torch.save(done, progress_path + ".tmp")
            os.replace(progress_path + ".tmp", progress_path)
        if log is not None:
            first = [h for h in rec["first_blowup_h"] if h is not None]
            log(f"[causes] {n + 1}/{len(names)} {name}: blow-ups {len(first)}"
                + (f" (first at {min(first)} h)" if first else ""))
    return summarize(done, [n for n in names if n in done], sc.head, steps, seed, len(indices))


def summarize(done: Dict[str, dict], names: Sequence[str], keys: Sequence[str], steps: int,
              seed: int = 0, inits: Optional[int] = None) -> Dict[str, object]:
    """Paired changes vs the intact model, interactions and the raw per-init records."""
    full = done["full"]
    ref_all = np.array(full["all"], np.float64)
    out: Dict[str, object] = {"leads_h": [6 * (l + 1) for l in range(steps)],
                              "pair_leads_h": [6 * s for s in _pair_steps(steps)], "keys": list(keys),
                              "inits": inits, "missing_edges": MISSING, "interventions": {}}
    rel_all = {}
    for name in ["full", *names]:
        rec = done[name]
        all_v = np.array(rec["all"], np.float64)
        s = {"edge": rec["edge"], "blowups": np.array(rec["blown"]).sum(0).tolist(),
             "first_blowup_h": rec["first_blowup_h"], "records": rec}
        if name != "full":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rel_all[name] = (all_v - ref_all) / np.nanmean(ref_all, 0)
            s["change_all"] = bootstrap(rel_all[name], seed)
            s["change"] = {}
            for k in keys:
                ref = np.array(full["rmse"][k], np.float64)
                s["change"][k] = bootstrap((np.array(rec["rmse"][k], np.float64) - ref) / np.nanmean(ref, 0), seed)
            s["response"] = {g: bootstrap(np.array(v, np.float64), seed) for g, v in rec["response"].items()}
        if "pair" in rec:
            a, b = rec["pair"]["a"], rec["pair"]["b"]
            p = {"a": a, "b": b}
            if a in rel_all and b in rel_all:
                p["interaction_all"] = bootstrap(rel_all[name] - rel_all[a] - rel_all[b], seed)
            for key in ("eta", "rho_a_given_b", "rho_b_given_a"):
                p[key] = {g: bootstrap(np.array(v, np.float64), seed) for g, v in rec["pair"][key].items()}
            s["pair"] = p
        out["interventions"][name] = s
    return out
