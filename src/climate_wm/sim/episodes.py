"""Stage 15 (simulator as teacher; 감자 18:42 KST 11/01): the SPEEDY episode dataset.

Each episode is one SPEEDY run from a nature-run restart, stored as its own
zarr with the WB2 state variables (as nature.zarr) and the per-process physics
tendencies on the same grid and levels (variables `tend_Q`, `tend_S`,
`tend_F_u`, `tend_F_v`, `tend_Q_cnv`, ...; units K/s, kg/kg/s, m/s^2; the mean
over the 6 h before each time). Groups:

  A  control: initial conditions only (different restart dates)
  B  perturbation: a spatially correlated perturbation of T, vorticity and log ps
     (the first `spinup_days` are spin-up)
  C  physics: convection time scale, surface drag, CO2 absorptivity, solar constant
  D  long rollout (stability)

The split is by episode (never by frame): `test_per_group` episodes of each
group are held out.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .speedy import (WB2_LEVELS, Intervention, Perturbation, SpeedyRunner, read_native, read_static,
                     read_tendencies, tendencies_to_wb2, to_dataset, to_wb2)

PERTURBATION = dict(t=0.5, vor=1.5e-6, ps=1e-3, scale=8.0)
CONV_RANGE = (0.5, 2.0)      # factor on the convective relaxation time, log-uniform
DRAG_RANGE = (0.5, 2.0)      # factor on the surface drag coefficients, log-uniform
CO2_RANGE = (0.8, 1.25)
SOLAR_RANGE = (0.98, 1.02)


@dataclasses.dataclass
class EpisodeSpec:
    name: str
    group: str
    start: dt.datetime
    days: int
    intervention: Intervention = dataclasses.field(default_factory=Intervention)
    perturbation: Optional[Perturbation] = None
    split: str = "train"
    spinup_days: int = 0

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["start"] = self.start.isoformat()
        return d


def plan_episodes(restart_dates: Sequence[dt.datetime], counts: Dict[str, int], days: int = 30,
                  long_days: int = 180, test_per_group: Optional[Dict[str, int]] = None,
                  seed: int = 0, spinup_days: int = 1) -> List[EpisodeSpec]:
    """Start dates spread evenly over the restarts (each group shifted, so the
    groups do not share dates), physics settings and perturbation seeds drawn
    from `seed`, and a per-group random held-out set."""
    rng = np.random.default_rng(seed)
    dates = sorted(restart_dates)
    test_per_group = test_per_group or {}
    specs: List[EpisodeSpec] = []
    for gi, (group, n) in enumerate(sorted(counts.items())):
        if n <= 0:
            continue
        length = long_days if group == "D" else days
        usable = [d for d in dates if d + dt.timedelta(days=length) <= dates[-1] + dt.timedelta(days=5)] \
            if group != "D" else dates
        idx = np.linspace(0, len(usable) - 1, n + 1)[:-1] + gi * len(usable) / (4.0 * max(n, 1))
        idx = np.clip(np.round(idx).astype(int), 0, len(usable) - 1)
        group_specs = []
        for i, k in enumerate(idx):
            iv, pert, spin = Intervention(), None, 0
            if group == "B":
                pert = Perturbation(seed=1000 + i + 100 * seed, **PERTURBATION)
                spin = spinup_days
            elif group == "C":
                iv = Intervention(conv_factor=float(np.exp(rng.uniform(*np.log(CONV_RANGE)))),
                                  drag_factor=float(np.exp(rng.uniform(*np.log(DRAG_RANGE)))),
                                  co2_factor=float(rng.uniform(*CO2_RANGE)),
                                  solar_factor=float(rng.uniform(*SOLAR_RANGE)))
            group_specs.append(EpisodeSpec(f"{group}{i:03d}", group, usable[k], length, iv, pert,
                                           spinup_days=spin))
        for j in rng.permutation(len(group_specs))[:test_per_group.get(group, 0)]:
            group_specs[j].split = "test"
        specs += group_specs
    return specs


def bitround(x: np.ndarray, keepbits: int) -> np.ndarray:
    """Round float32 mantissas to `keepbits` bits (round half to even), so the
    zeroed low bits compress; relative error <= 2^-(keepbits+1)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    drop = 23 - keepbits
    if drop <= 0:
        return x
    b = x.view(np.uint32).astype(np.uint64)
    half = np.uint64(1 << (drop - 1))
    b = (b + half - np.uint64(1) + ((b >> np.uint64(drop)) & np.uint64(1))) & np.uint64(
        (0xFFFFFFFF >> drop) << drop)
    return b.astype(np.uint32).view(np.float32)


def write_episode(runner: SpeedyRunner, spec: EpisodeSpec, nature_root: str, out_dir: str,
                  levels: Sequence[int] = WB2_LEVELS, keepbits_state: int = 12, keepbits_tend: int = 7,
                  keep_run: bool = False) -> Path:
    """Runs one episode (with the per-process tendencies) and writes OUT/<name>.zarr."""
    import zarr
    root = Path(nature_root)
    out = Path(out_dir) / f"{spec.name}.zarr"
    if out.exists() and (out / "zarr.json").exists() and json.loads((out / "zarr.json").read_text()).get(
            "attributes", {}).get("complete"):
        return out
    if out.exists():
        shutil.rmtree(out)
    run = runner.run(spec.name, spec.start, spec.start + dt.timedelta(days=spec.days),
                     restart_in=str(root / "restarts" / f"{spec.start:%Y%m%d}.bin"), restart_out=False,
                     intervention=spec.intervention, perturbation=spec.perturbation)
    times = runner.output_times(run, spec.start)
    native = read_native(run, times)
    static = read_static(root / "static.nc")
    arrays = to_wb2(native, static, levels)
    tend = tendencies_to_wb2(read_tendencies(run, times), native, levels)
    for k, v in arrays.items():
        if v.ndim >= 3:
            arrays[k] = bitround(v, keepbits_state)
    for k, v in tend.items():
        if k not in ("Q", "S"):   # the totals are sums of the stored processes (episode_tendencies)
            arrays[f"tend_{k}"] = bitround(v, keepbits_tend)
    attrs = {"source": "SPEEDY speedy.f90 T30 episode (stage 15)", "spec": json.dumps(spec.to_json()),
             "complete": False}
    ds = to_dataset(arrays, times, levels, attrs=attrs)
    codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    enc = {v: {"compressors": [codec]} for v in ds.data_vars}
    ds.chunk({"time": len(times)}).to_zarr(out, encoding=enc, mode="w")
    meta = json.loads((out / "zarr.json").read_text())
    meta["attributes"]["complete"] = True
    (out / "zarr.json").write_text(json.dumps(meta))
    if not keep_run:
        shutil.rmtree(run, ignore_errors=True)
    return out


def episode_tendencies(ds) -> Dict[str, "object"]:
    """The column targets of an episode dataset: Q (K/s), S (kg/kg/s), F_u, F_v (m/s^2)
    as xarray DataArrays [time, level, longitude, latitude], and the processes."""
    q = sum(ds[f"tend_Q_{p}"] for p in ("cnv", "lsc", "sw", "lw", "pbl"))
    s = sum(ds[f"tend_S_{p}"] for p in ("cnv", "lsc", "pbl"))
    return {"Q": q, "S": s, "F_u": ds["tend_F_u"], "F_v": ds["tend_F_v"],
            **{k[5:]: ds[k] for k in ds.data_vars if k.startswith("tend_")}}


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
