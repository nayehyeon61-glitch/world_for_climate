"""SPEEDY (speedy.f90, T30, 8 sigma levels) as an environment for the world model.

Stage 14: the simulator is the "world" that reacts to actions. This module

* runs SPEEDY segments with restart files and interventions (`Intervention`),
* reads the 6-hourly output (upper air: u, v, t, q, geopotential height, ps on
  the 96x48 Gaussian grid; surface: accumulated precipitation, near-surface
  t/u/v, skin temperature, SST, insolation, OLR),
* maps it to the WeatherBench2 64x32 / 13-level layout the world models read
  (`to_wb2`): sigma to pressure levels (log-p interpolation, lapse-rate
  extrapolation below ground), omega from the sigma-coordinate continuity
  equation, mean sea level pressure, bilinear regridding.

The patched SPEEDY build (restart, interventions, surface output) comes from
`scripts/setup_speedy.sh` and `sim/speedy_env.patch` (this package).
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

WB2_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
WB2_LAT = np.linspace(-87.1875, 87.1875, 32)
WB2_LON = np.arange(64) * 5.625
# SPEEDY constants (physical_constants.f90) and half sigma levels (geometry.f90)
GRAV = 9.81
R_D = 287.0
GAMMA = 6.5e-3          # K/m, standard lapse rate for extrapolation below ground
EARTH_RADIUS = 6.371e6
SIGMA_HALF = np.array([0.0, 0.05, 0.14, 0.26, 0.42, 0.6, 0.77, 0.9, 1.0])
BC_FILES = {"clim": ("surface.nc", "sea_surface_temperature.nc", "sea_ice.nc", "land.nc",
                     "snow.nc", "soil.nc"),
            "anom": ("sea_surface_temperature_anomaly.nc",)}


@dataclasses.dataclass
class Intervention:
    """Actions of the experimenter (the `&control` namelist of the patched SPEEDY)."""
    solar_factor: float = 1.0
    co2_factor: float = 1.0
    conv_factor: float = 1.0
    drag_factor: float = 1.0
    # SST patches: (lon deg E, lat deg N, e-folding radius deg, amplitude K)
    patches: Tuple[Tuple[float, float, float, float], ...] = ()

    def namelist(self) -> str:
        lines = [f"solar_factor = {self.solar_factor}", f"co2_factor = {self.co2_factor}",
                 f"conv_factor = {self.conv_factor}", f"drag_factor = {self.drag_factor}",
                 f"n_patch = {len(self.patches)}"]
        for i, (lon, lat, rad, amp) in enumerate(self.patches, 1):
            lines += [f"patch_lon({i}) = {lon}", f"patch_lat({i}) = {lat}",
                      f"patch_radius({i}) = {rad}", f"patch_amp({i}) = {amp}"]
        return "\n".join(lines)

    def is_null(self) -> bool:
        return self == Intervention()


@dataclasses.dataclass
class Perturbation:
    """Stage 15 episode group B: a smooth random perturbation of the restart state
    (red spectrum with e-folding total wavenumber `scale`, zero global mean)."""
    seed: int = 0          # 0 = none
    t: float = 0.0         # rms K
    vor: float = 0.0       # rms 1/s
    ps: float = 0.0        # rms of log surface pressure
    scale: float = 8.0

    def namelist(self) -> str:
        return (f"perturb_seed = {self.seed}\nperturb_t = {self.t}\nperturb_vor = {self.vor}\n"
                f"perturb_ps = {self.ps}\nperturb_scale = {self.scale}")


def _dt_block(name: str, t: dt.datetime) -> str:
    return "\n".join(f"{name}%{k} = {getattr(t, k)}"
                     for k in ("year", "month", "day", "hour", "minute"))


class SpeedyRunner:
    """Runs the patched SPEEDY binary in its own directory per segment."""

    def __init__(self, root: str, workdir: str, nsteps_out: int = 9, tendencies: bool = False):
        self.root = Path(root)
        self.tendencies = tendencies   # stage 15: also write yyyymmddhhmm_tend.nc (per-process physics)
        self.binary = self.root / "bin" / "speedy"
        self.bc = self.root / "data" / "bc" / "t30"
        self.workdir = Path(workdir)
        self.nsteps_out = nsteps_out   # 36 steps per day: 9 = 6-hourly output
        if not self.binary.exists():
            raise FileNotFoundError(f"{self.binary}: run scripts/setup_speedy.sh first")

    def run(self, tag: str, start: dt.datetime, end: dt.datetime,
            restart_in: Optional[str] = None, restart_out: bool = True,
            intervention: Optional[Intervention] = None, static: bool = False,
            perturbation: Optional["Perturbation"] = None) -> Path:
        """One segment from `start` to `end` (00 UTC boundaries keep the daily
        forcing and shortwave cycles aligned with a straight run)."""
        if start.hour or start.minute:
            raise ValueError("segments must start at 00 UTC")
        run = self.workdir / tag
        run.mkdir(parents=True, exist_ok=True)
        for f in run.glob("*.nc"):
            f.unlink()
        for sub, names in BC_FILES.items():
            for n in names:
                link = run / n
                if not link.exists():
                    link.symlink_to(self.bc / sub / n)
        iv = intervention or Intervention()
        control = [f"restart_in = '{os.path.abspath(restart_in) if restart_in else ''}'",
                   f"restart_out = '{run / 'restart.bin' if restart_out else ''}'",
                   f"static_out = '{run / 'static.nc' if static else ''}'", iv.namelist()]
        if self.tendencies:
            control.append("tend_out = .true.")
        if perturbation is not None and perturbation.seed:
            if not restart_in:
                raise ValueError("a perturbation needs a restart to perturb")
            control.append(perturbation.namelist())
        text = ("&params\nnsteps_out = %d\nnstdia = 100000\n/\n&date\n%s\n%s\n/\n&control\n%s\n/\n"
                % (self.nsteps_out, _dt_block("start_datetime", start),
                   _dt_block("end_datetime", end), "\n".join(control)))
        (run / "namelist.nml").write_text(text)
        res = subprocess.run([str(self.binary)], cwd=run, capture_output=True, text=True)
        (run / "speedy.log").write_text(res.stdout + res.stderr)
        if res.returncode != 0 or "STOP" in res.stdout + res.stderr:
            raise RuntimeError(f"SPEEDY failed in {run}: {(res.stdout + res.stderr)[-800:]}")
        return run

    @staticmethod
    def output_times(run: Path, start: dt.datetime) -> List[dt.datetime]:
        """Valid output times of a segment, excluding the start (whose upper-air
        file is the rest state when a restart is read)."""
        times = []
        for f in sorted(run.glob("*_sfc.nc")):
            t = dt.datetime.strptime(f.name[:12], "%Y%m%d%H%M")
            if t > start:
                times.append(t)
        return times


def read_native(run: Path, times: Sequence[dt.datetime]) -> Dict[str, np.ndarray]:
    """Stacks the native output: upper-air [T, K, 48, 96], surface [T, 48, 96]."""
    import netCDF4
    out: Dict[str, List[np.ndarray]] = {}
    for t in times:
        stamp = t.strftime("%Y%m%d%H%M")
        with netCDF4.Dataset(run / f"{stamp}.nc") as ds:
            for v in ("u", "v", "t", "q", "phi", "ps"):
                out.setdefault(v, []).append(np.asarray(ds[v][0], dtype=np.float64))
            lat = np.asarray(ds["lat"][:], dtype=np.float64)
            sigma = np.asarray(ds["lev"][:], dtype=np.float64)
        with netCDF4.Dataset(run / f"{stamp}_sfc.nc") as ds:
            for v in ("tp", "t2m", "u10", "v10", "skt", "sst", "tisr", "olr", "tsr"):
                out.setdefault(v, []).append(np.asarray(ds[v][0], dtype=np.float64))
    arrays = {k: np.stack(v) for k, v in out.items()}
    arrays["lat"] = lat
    arrays["sigma"] = sigma
    return arrays


def read_static(path: Path) -> Dict[str, np.ndarray]:
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        return {k: np.asarray(ds[k][:], dtype=np.float64)
                for k in ("orography", "land_fraction", "sea_fraction", "lat")}


# --------------------------------------------------------------------------- physics mapping
def _ddx(f: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """d/dlambda (periodic, centred) and d/dphi (np.gradient on the Gaussian
    latitudes) of a field [..., lat, lon]."""
    nlon = f.shape[-1]
    dlam = 2 * np.pi / nlon
    dfl = (np.roll(f, -1, -1) - np.roll(f, 1, -1)) / (2 * dlam)
    dfp = np.gradient(f, np.deg2rad(lat), axis=-2)
    return dfl, dfp


def omega_sigma(u: np.ndarray, v: np.ndarray, ps: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Pressure velocity (Pa/s) at full sigma levels from the continuity equation
    omega = sigma (dps/dt + V.grad ps) - int_0^sigma div(ps V) dsigma'.
    u, v [T, K, lat, lon] (m/s), ps [T, lat, lon] (Pa)."""
    coslat = np.cos(np.deg2rad(lat))[:, None]
    a = EARTH_RADIUS
    dsig = np.diff(SIGMA_HALF)                      # [K]
    sig = 0.5 * (SIGMA_HALF[1:] + SIGMA_HALF[:-1])
    pu = ps[:, None] * u
    pvc = ps[:, None] * v * coslat
    dpu, _ = _ddx(pu, lat)
    _, dpv = _ddx(pvc, lat)
    div = (dpu + dpv) / (a * coslat)                # div(ps V) [T, K, lat, lon]
    dpl, dpp = _ddx(ps, lat)
    vgrad = (u * dpl[:, None] / coslat + v * dpp[:, None]) / a
    dps_dt = -(div * dsig[None, :, None, None]).sum(1)
    above = np.cumsum(div * dsig[None, :, None, None], axis=1) - 0.5 * div * dsig[None, :, None, None]
    return sig[None, :, None, None] * (dps_dt[:, None] + vgrad) - above


def to_pressure(x: np.ndarray, ps: np.ndarray, sigma: np.ndarray, levels_hpa: Sequence[int],
                kind: str, t_low: Optional[np.ndarray] = None) -> np.ndarray:
    """Sigma levels [T, K, ...] to pressure levels [T, L, ...], linear in log p.
    Below the lowest model level: temperature with the standard lapse rate,
    geopotential height hydrostatically, other fields held constant."""
    K = x.shape[1]
    logp = np.log(sigma[None, :, None, None] * ps[:, None])        # increasing with k
    out = np.empty((x.shape[0], len(levels_hpa)) + x.shape[2:])
    expo = R_D * GAMMA / GRAV
    for li, lev in enumerate(levels_hpa):
        lp = np.log(lev * 100.0)
        n = (logp <= lp).sum(1)                                     # [T, lat, lon]
        k0 = np.clip(n - 1, 0, K - 2)[:, None]
        l0 = np.take_along_axis(logp, k0, 1)[:, 0]
        l1 = np.take_along_axis(logp, k0 + 1, 1)[:, 0]
        x0 = np.take_along_axis(x, k0, 1)[:, 0]
        x1 = np.take_along_axis(x, k0 + 1, 1)[:, 0]
        w = np.clip((lp - l0) / (l1 - l0), 0.0, 1.0)
        val = x0 + w * (x1 - x0)
        below = lp > logp[:, -1]
        if below.any():
            ratio = np.exp(lp - logp[:, -1])                        # p / p_K > 1
            if kind == "t":
                ext = x[:, -1] * ratio ** expo
            elif kind == "z":
                ext = x[:, -1] + t_low / GAMMA * (1.0 - ratio ** expo)
            else:
                ext = x[:, -1]
            val = np.where(below, ext, val)
        out[:, li] = val
    return out


def mean_sea_level_pressure(ps: np.ndarray, t_low: np.ndarray, p_low: np.ndarray,
                            orog: np.ndarray) -> np.ndarray:
    """Reduction with the standard lapse rate from the lowest-level temperature."""
    expo = R_D * GAMMA / GRAV
    ts = t_low * (ps / p_low) ** expo
    return ps * (1.0 + GAMMA * np.maximum(orog, 0.0) / ts) ** (1.0 / expo)


class Regridder:
    """Bilinear map from the SPEEDY Gaussian grid (lat S->N, lon from 0) to the
    WB2 64x32 equiangular grid (edge latitudes clamped)."""

    def __init__(self, src_lat: np.ndarray, nlon_src: int = 96,
                 dst_lat: np.ndarray = WB2_LAT, dst_lon: np.ndarray = WB2_LON):
        fi = np.interp(dst_lat, src_lat, np.arange(len(src_lat)))
        self.j0 = np.floor(fi).astype(int).clip(0, len(src_lat) - 2)
        self.wj = (fi - self.j0)[:, None]
        fl = dst_lon / (360.0 / nlon_src)
        self.i0 = np.floor(fl).astype(int) % nlon_src
        self.i1 = (self.i0 + 1) % nlon_src
        self.wi = (fl - np.floor(fl))[None, :]

    def __call__(self, f: np.ndarray) -> np.ndarray:
        a = f[..., self.j0, :]
        b = f[..., self.j0 + 1, :]
        rows = a + self.wj * (b - a)
        return rows[..., self.i0] * (1 - self.wi) + rows[..., self.i1] * self.wi


def to_wb2(native: Dict[str, np.ndarray], static: Dict[str, np.ndarray],
           levels: Sequence[int] = WB2_LEVELS) -> Dict[str, np.ndarray]:
    """WB2 variables on (time, [level,] longitude, latitude), float32, SI units
    of the WB2 archive (geopotential m^2/s^2, tp m per 6 h, tisr J/m^2 per hour)."""
    lat, sigma, ps = native["lat"], native["sigma"], native["ps"]
    t_low = native["t"][:, -1]
    p_low = sigma[-1] * ps
    omega = omega_sigma(native["u"], native["v"], ps, lat)
    plev = {
        "temperature": to_pressure(native["t"], ps, sigma, levels, "t"),
        "geopotential": to_pressure(native["phi"], ps, sigma, levels, "z", t_low=t_low) * GRAV,
        "u_component_of_wind": to_pressure(native["u"], ps, sigma, levels, "u"),
        "v_component_of_wind": to_pressure(native["v"], ps, sigma, levels, "v"),
        "specific_humidity": np.maximum(to_pressure(native["q"], ps, sigma, levels, "q"), 0.0),
        "vertical_velocity": to_pressure(omega, ps, sigma, levels, "w"),
    }
    sfc = {
        "2m_temperature": native["t2m"],
        "10m_u_component_of_wind": native["u10"],
        "10m_v_component_of_wind": native["v10"],
        "mean_sea_level_pressure": mean_sea_level_pressure(ps, t_low, p_low, static["orography"]),
        "total_precipitation_6hr": native["tp"] * 1e-3,
        "toa_incident_solar_radiation": native["tisr"] * 3600.0,
        "sea_surface_temperature": native["sst"],
    }
    rg = Regridder(lat, nlon_src=native["ps"].shape[-1])
    out = {}
    for k, v in {**plev, **sfc}.items():
        out[k] = np.swapaxes(rg(v), -1, -2).astype(np.float32)     # [..., lon, lat]
    out["geopotential_at_surface"] = np.swapaxes(rg(static["orography"] * GRAV), -1, -2).astype(np.float32)
    out["land_sea_mask"] = np.swapaxes(rg(static["land_fraction"]), -1, -2).astype(np.float32)
    return out


def to_dataset(arrays: Dict[str, np.ndarray], times: Sequence[dt.datetime],
               levels: Sequence[int] = WB2_LEVELS, attrs: Optional[dict] = None):
    """xarray Dataset with the WB2 dimension names and order."""
    import xarray as xr
    coords = {"time": np.array(times, dtype="datetime64[ns]"), "level": np.array(levels),
              "longitude": WB2_LON, "latitude": WB2_LAT}
    data = {}
    for k, v in arrays.items():
        if v.ndim == 4:
            data[k] = (("time", "level", "longitude", "latitude"), v)
        elif v.ndim == 3:
            data[k] = (("time", "longitude", "latitude"), v)
        else:
            data[k] = (("longitude", "latitude"), v)
    return xr.Dataset(data, coords=coords, attrs=attrs or {})


# --------------------------------------------------------------------------- stage 15: teacher tendencies
TEND_NAMES = ("tt_cnv", "tt_lsc", "tt_rsw", "tt_rlw", "tt_pbl", "qt_cnv", "qt_lsc", "qt_pbl", "ut_pbl", "vt_pbl")
# the per-process tendencies on pressure levels: N1 column outputs Q (K/s), S (kg/kg/s), F (m/s^2)
TEND_GROUPS = {
    "Q_cnv": ("tt_cnv",), "Q_lsc": ("tt_lsc",), "Q_sw": ("tt_rsw",), "Q_lw": ("tt_rlw",), "Q_pbl": ("tt_pbl",),
    "S_cnv": ("qt_cnv",), "S_lsc": ("qt_lsc",), "S_pbl": ("qt_pbl",),
    "F_u": ("ut_pbl",), "F_v": ("vt_pbl",),
}


def read_tendencies(run: Path, times: Sequence[dt.datetime]) -> Dict[str, np.ndarray]:
    """Mean per-process physics tendencies over the 6 h before each output time,
    [T, K, 48, 96] on the model's sigma levels (T: K/s, q: g/kg/s, u, v: m/s^2)."""
    import netCDF4
    out: Dict[str, List[np.ndarray]] = {}
    for t in times:
        with netCDF4.Dataset(run / f"{t.strftime('%Y%m%d%H%M')}_tend.nc") as ds:
            for v in TEND_NAMES:
                out.setdefault(v, []).append(np.asarray(ds[v][0], dtype=np.float64))
    return {k: np.stack(v) for k, v in out.items()}


def tendencies_to_wb2(tend: Dict[str, np.ndarray], native: Dict[str, np.ndarray],
                      levels: Sequence[int] = WB2_LEVELS) -> Dict[str, np.ndarray]:
    """Per-process tendencies on the WB2 grid and pressure levels, [T, L, lon, lat] float32:
    Q_* (K/s), S_* (kg/kg/s; SPEEDY q is g/kg), F_u, F_v (m/s^2), and the totals Q, S.
    Zero below the surface (p > ps); between the lowest full level and ps held constant.
    The pressure of each column is the one at the end of the averaging interval."""
    lat, sigma, ps = native["lat"], native["sigma"], native["ps"]
    rg = Regridder(lat, nlon_src=ps.shape[-1])
    above = (np.asarray(levels, dtype=np.float64)[None, :, None, None] * 100.0) <= ps[:, None]
    out = {}
    for name, srcs in TEND_GROUPS.items():
        x = sum(tend[s] for s in srcs)
        if name.startswith("S_"):
            x = x * 1e-3
        xp = to_pressure(x, ps, sigma, levels, "tend") * above
        out[name] = np.swapaxes(rg(xp), -1, -2).astype(np.float32)
    out["Q"] = sum(out[k] for k in ("Q_cnv", "Q_lsc", "Q_sw", "Q_lw", "Q_pbl"))
    out["S"] = sum(out[k] for k in ("S_cnv", "S_lsc", "S_pbl"))
    return out
