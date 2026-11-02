"""GraphCast forcings: year/day progress and TOA incident solar radiation.

`year_progress`/`day_progress` reproduce `weathernext/utils/data_utils.py`.
`toa_incident_solar_radiation` is a NumPy port of
`weathernext/utils/solar_radiation.py` (ERA5-compatible orbital formulas,
1h integration ending at the timestamp). Prefer ERA5's own `tisr` when it is
in the archive; the computed version is for target times / synthetic data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

SEC_PER_DAY = 86400
AVG_DAY_PER_YEAR = 365.24219
_J2000_EPOCH = 2451545.0
_JULIAN_YEAR_LENGTH_IN_DAYS = 365.25
_REFERENCE_TSI = 1361.0
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # trapz before NumPy 2.0


def seconds_since_epoch(times) -> np.ndarray:
    return np.asarray(times, dtype="datetime64[s]").astype(np.int64)


def year_progress(times) -> np.ndarray:
    """[T] in [0, 1)."""
    years = seconds_since_epoch(times) / SEC_PER_DAY / np.float64(AVG_DAY_PER_YEAR)
    return np.mod(years, 1.0).astype(np.float32)


def day_progress(times, lon_deg: np.ndarray) -> np.ndarray:
    """[T, num_lon] in [0, 1): local solar-ish time per longitude."""
    greenwich = np.mod(seconds_since_epoch(times), SEC_PER_DAY) / SEC_PER_DAY
    offsets = np.deg2rad(lon_deg) / (2 * np.pi)
    return np.mod(greenwich[:, None] + offsets[None, :], 1.0).astype(np.float32)


def _orbital(j2000_days: np.ndarray):
    theta = j2000_days / _JULIAN_YEAR_LENGTH_IN_DAYS
    rotational_phase = j2000_days % 1.0
    rel = 1.7535 + 6.283076 * theta
    rem = 6.240041 + 6.283020 * theta
    rlls = 4.8951 + 6.283076 * theta
    one = np.ones_like(theta)
    rllls = np.stack([one, theta, np.sin(rel), np.cos(rel), np.sin(2 * rel),
                      np.cos(2 * rel)], -1) @ np.array(
                          [4.8952, 6.283320, -0.0075, -0.0326, -0.0003, 0.0002])
    sin_decl = np.sin(0.409093) * np.sin(rllls)
    cos_decl = np.sqrt(1.0 - sin_decl ** 2)
    eq_time = np.stack([np.sin(2 * rlls), np.sin(rem),
                        np.sin(rem) * np.cos(2 * rlls), np.sin(4 * rlls),
                        np.sin(2 * rem)], -1) @ np.array(
                            [591.8, -459.4, 39.5, -12.7, -4.8])
    distance = np.stack([one, np.sin(rel), np.cos(rel)], -1) @ np.array(
        [1.0001, -0.0163, 0.0037])
    return rotational_phase, sin_decl, cos_decl, eq_time, distance


def toa_incident_solar_radiation(times: Sequence, lat_deg: np.ndarray,
                                 lon_deg: np.ndarray, num_bins: int = 12,
                                 tsi: float = _REFERENCE_TSI) -> np.ndarray:
    """TOA incident solar radiation integrated over the preceding hour [J/m^2].

    lat [H], lon [W] -> [T, H, W] (trapezoidal rule with `num_bins` bins;
    weathernext uses 360 bins, 12 is within ~0.1% for hourly integrals).
    """
    lat = np.deg2rad(np.asarray(lat_deg, np.float64))[:, None]
    lon = np.deg2rad(np.asarray(lon_deg, np.float64))[None, :]
    return _integrate(times, np.sin(lat), np.cos(lat), lon, num_bins, tsi)


def _integrate(times, sin_lat, cos_lat, lon, num_bins, tsi):
    offsets = np.linspace(-1.0 / 24.0, 0.0, num_bins + 1)
    dx = 3600.0 / num_bins
    out = []
    for t in times:
        days = pd.Timestamp(t).to_julian_date() - _J2000_EPOCH + offsets
        phase, sin_d, cos_d, eq_time, dist = _orbital(days)
        solar_time = phase + eq_time / SEC_PER_DAY
        flux = []
        for k in range(len(days)):
            hour_angle = 2.0 * np.pi * solar_time[k] + lon
            sin_alt = cos_lat * cos_d[k] * np.cos(hour_angle) + sin_lat * sin_d[k]
            flux.append(tsi / dist[k] ** 2 * np.maximum(sin_alt, 0.0))
        out.append(_trapezoid(np.stack(flux, -1), dx=dx, axis=-1))
    return np.stack(out).astype(np.float32)


def time_forcings(times, lon_deg: np.ndarray, num_lat: int) -> dict:
    """year/day progress sin/cos broadcast to [T, H, W] grids."""
    yp = year_progress(times)[:, None, None] * 2 * np.pi
    dp = day_progress(times, lon_deg)[:, None, :] * 2 * np.pi
    shape = (len(yp), num_lat, len(lon_deg))
    return {
        "year_progress_sin": np.broadcast_to(np.sin(yp), shape).astype(np.float32),
        "year_progress_cos": np.broadcast_to(np.cos(yp), shape).astype(np.float32),
        "day_progress_sin": np.broadcast_to(np.sin(dp), shape).astype(np.float32),
        "day_progress_cos": np.broadcast_to(np.cos(dp), shape).astype(np.float32),
    }
