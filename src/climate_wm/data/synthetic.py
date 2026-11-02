"""Synthetic ERA5-like data for P0 smoke tests (NOT a physical simulation).

Fields use GraphCast names / units and roughly realistic magnitudes. The
dynamics are deliberately Lie-structured plus residual:

  * every pressure shell carries wave patterns that rotate rigidly about the
    Earth's axis with a level-dependent angular speed (an so(3) action about
    z, faster aloft = vertical shear), with winds consistent with the rotation;
  * the wave amplitudes evolve slowly and a diurnal cycle follows the computed
    TOA radiation (non-rigid "residual" dynamics);
  * small noise.

So a model with a working Lie generator should learn omega ~ Omega_level z.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import forcings as forcing_lib

EARTH_RADIUS_M = 6.371e6
G = 9.80665


def _height_m(level_hpa: np.ndarray) -> np.ndarray:
    return 7400.0 * np.log(1013.25 / level_hpa)


def make_synthetic_era5(num_days: int = 30, num_lat: int = 33, num_lon: int = 64,
                        levels: Sequence[int] = (850, 500, 250),
                        start: str = "2016-01-01", seed: int = 0,
                        include_vertical_velocity: bool = False,
                        noise: float = 0.02) -> xr.Dataset:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, periods=num_days * 4, freq="6h").values
    lat = np.linspace(-90, 90, num_lat)
    lon = np.arange(num_lon) * 360.0 / num_lon
    levels = np.array(sorted(levels), dtype=np.int64)
    t_hours = (times - times[0]) / np.timedelta64(1, "h")

    phi = np.deg2rad(lat)[None, None, :, None]
    lam = np.deg2rad(lon)[None, None, None, :]
    th = t_hours[:, None, None, None]
    # Angular speed (rad per hour) per level: ~5 m/s at 850 hPa .. ~25 m/s aloft.
    speed_ms = 5.0 + 20.0 * (1 - levels / 1000.0) ** 1.5
    omega = (speed_ms / EARTH_RADIUS_M * 3600.0)[None, :, None, None]
    modes = [(3, 1.0, 0.3), (5, 0.6, 1.7), (7, 0.35, 4.0)]
    pattern = np.zeros((len(times), len(levels), num_lat, num_lon))
    dpattern_dlam = np.zeros_like(pattern)
    for m, amp, phase in modes:
        a = amp * (1 + 0.3 * np.sin(2 * np.pi * th / (24 * 9.0) + m))
        arg = m * (lam - omega * th) + phase
        env = np.cos(phi) ** 2 * (1 + 0.5 * np.sin(2 * phi))
        pattern += a * env * np.cos(arg)
        dpattern_dlam += -m * a * env * np.sin(arg)
    pattern += noise * rng.standard_normal(pattern.shape)

    lev = levels[None, :, None, None].astype(np.float64)
    height = _height_m(lev)
    temperature = 288.0 - 6.5e-3 * np.minimum(height, 11000) - 35 * np.sin(phi) ** 2 + 4.0 * pattern
    geopotential = G * height - 600.0 * np.sin(phi) ** 2 * (lev / 500.0) + 400.0 * pattern
    u_wind = (omega / 3600.0) * EARTH_RADIUS_M * np.cos(phi) + 1.5 * pattern
    v_wind = 3.0 * dpattern_dlam / np.maximum(np.cos(phi), 0.05) * np.cos(phi) ** 2
    q = 0.012 * (lev / 1000.0) ** 3 * np.exp(-3 * np.sin(phi) ** 2) * (1 + 0.1 * pattern)

    tisr = forcing_lib.toa_incident_solar_radiation(times, lat, lon)
    diurnal = tisr / 4.9e6  # ~[0, 1]
    surf = pattern[:, -1]  # lowest level (largest pressure is last after sort)
    t2m = temperature[:, -1] + 2.0 + 3.0 * diurnal
    msl = 101325.0 - 1500.0 * np.sin(2 * np.deg2rad(lat))[None, :, None] ** 2 + 800.0 * surf
    u10 = 0.7 * u_wind[:, -1]
    v10 = 0.7 * v_wind[:, -1]
    tp6 = 1e-3 * np.maximum(surf, 0.0) ** 2 * (1 + diurnal)

    lon2d, lat2d = np.meshgrid(lon, lat)
    orography = G * 800.0 * np.maximum(0.0, np.sin(3 * np.deg2rad(lon2d)) * np.cos(2 * np.deg2rad(lat2d)))
    lsm = (orography > 0).astype(np.float32)

    data = {
        "temperature": (("time", "level", "lat", "lon"), temperature),
        "geopotential": (("time", "level", "lat", "lon"), geopotential),
        "u_component_of_wind": (("time", "level", "lat", "lon"), u_wind),
        "v_component_of_wind": (("time", "level", "lat", "lon"), v_wind),
        "specific_humidity": (("time", "level", "lat", "lon"), q),
        "2m_temperature": (("time", "lat", "lon"), t2m),
        "mean_sea_level_pressure": (("time", "lat", "lon"), msl),
        "10m_u_component_of_wind": (("time", "lat", "lon"), u10),
        "10m_v_component_of_wind": (("time", "lat", "lon"), v10),
        "total_precipitation_6hr": (("time", "lat", "lon"), tp6),
        "toa_incident_solar_radiation": (("time", "lat", "lon"), tisr),
        "geopotential_at_surface": (("lat", "lon"), orography),
        "land_sea_mask": (("lat", "lon"), lsm),
    }
    if include_vertical_velocity:
        data["vertical_velocity"] = (("time", "level", "lat", "lon"), 0.1 * pattern)
    ds = xr.Dataset(
        {k: (dims, np.asarray(v, np.float32)) for k, (dims, v) in data.items()},
        coords={"time": times, "level": levels, "lat": lat, "lon": lon})
    ds.attrs["description"] = "synthetic ERA5-like smoke-test data"
    return ds
