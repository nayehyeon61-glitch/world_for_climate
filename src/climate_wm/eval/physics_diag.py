"""Differentiable physical diagnostics of a GraphCast-layout state (stage 5).

The stage 5 GNN receives information about how "physical" the observed state
is, and the critic judges predicted states with the same quantities. All
diagnostics are computed from the unnormalised grid fields (lat ascending,
lon periodic) with spherical finite differences, so they are differentiable
in the model's predictions:

  hydrostatic   (Phi_up - Phi_lo) / (R_d Tv ln(p_lo / p_up)) - 1 per adjacent
                level pair (relative thickness error; Tv = T (1 + 0.608 q))
  geostrophic   |f k x v + grad Phi| / (|f| |v| + |grad Phi|) in [0, 1]
                (0 = balanced), per level and at the surface with
                grad(msl) / rho_0; zero within `equator_mask_deg` of the
                equator where f -> 0 (and on pole rows)
  stability     N^2 = g^2 (theta_up - theta_lo) / (theta * R_d Tv ln(p_lo / p_up))
                per level pair (the thickness replaces dz so that a predicted
                inverted geopotential cannot divide by zero)
  saturation    1 - q / q_s(T, p) (Bolton's e_s over water); <= 0 is saturated
  kinematics    horizontal divergence and relative vorticity per level and of
                the 10 m wind
  surface       2 m temperature - T(1000 hPa), a surface-budget proxy: WB2's
                graphcast_13 data has no radiation fluxes or skin temperature
                (the Lustre archive has both, where skt - t2m is the direct form)

The diurnal phase is not repeated here: the `day_progress` forcings are the
local solar time per longitude. Pair diagnostics are assigned to the lower
level of the pair, so each lands in the shell of that level. As encoder
inputs every diagnostic is squashed by tanh(x / scale) with a fixed physical
scale; `raw` returns the unsquashed values for the critic.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ..data.graphcast_mapping import Channel, ChannelLayout

R_D = 287.05  # J kg^-1 K^-1
CP = 1004.6
KAPPA = R_D / CP
GRAVITY = 9.80665
EARTH_RADIUS = 6.371e6
EARTH_OMEGA = 7.292e-5
P_REF = 1.0e5  # Pa, potential temperature reference
RHO_SURFACE = 1.2  # kg m^-3, for the surface pressure gradient

FAMILIES = ("hydrostatic", "geostrophic", "stability", "saturation", "kinematics", "surface")
# Squashing scales of the encoder features (typical magnitudes on a 5.6 deg grid).
SCALES = {"hydrostatic": 0.01, "geostrophic": 1.0, "stability": 2e-4, "saturation": 1.0,
          "divergence": 1e-5, "vorticity": 3e-5, "surface": 5.0}


@dataclasses.dataclass
class DiagnosticsConfig:
    enabled: bool = False
    families: Tuple[str, ...] = FAMILIES
    equator_mask_deg: float = 10.0

    def __post_init__(self):
        self.families = tuple(self.families)
        unknown = set(self.families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown diagnostic families {sorted(unknown)}")


def with_diagnostics(layout: ChannelLayout, channels: Sequence[Channel]) -> ChannelLayout:
    """Copy of `layout` whose encoder input is [dynamic | diagnostics | forcings | static]."""
    out = copy.copy(layout)
    out.diagnostics = list(channels)
    out.diagnostic_shells = layout.partition(out.diagnostics)
    return out


class PhysicsDiagnostics(nn.Module):

    def __init__(self, layout: ChannelLayout, lat_deg: np.ndarray, lon_deg: np.ndarray,
                 cfg: DiagnosticsConfig):
        super().__init__()
        self.cfg = cfg
        self.layout = layout
        index = {c: i for i, c in enumerate(layout.dynamic)}
        self._index = index
        levels = sorted(layout.task.pressure_levels, reverse=True)  # bottom -> top
        has = lambda name, lvl=None: Channel(name, lvl) in index
        self.levels = [l for l in levels if all(has(n, l) for n in (
            "temperature", "geopotential", "u_component_of_wind", "v_component_of_wind"))]
        self.humid = all(has("specific_humidity", l) for l in self.levels)
        self.pairs = list(zip(self.levels[:-1], self.levels[1:]))  # (lower, upper)
        self.surface_wind = has("10m_u_component_of_wind") and has("10m_v_component_of_wind")
        self.surface_msl = has("mean_sea_level_pressure")
        self.surface_t2m = has("2m_temperature") and 1000 in self.levels

        lon = np.asarray(lon_deg, np.float64)
        dlon = np.diff(lon)
        if len(lon) < 3 or not np.allclose(dlon, dlon[0]) or not np.isclose(dlon[0] * len(lon), 360.0):
            raise ValueError("diagnostics need a uniform, periodic longitude grid")
        lat = np.deg2rad(np.asarray(lat_deg, np.float64))
        if np.any(np.diff(lat) <= 0):
            raise ValueError("diagnostics need ascending latitudes")
        self.dlon = math.radians(float(dlon[0]))
        # Pole rows (0.25 deg ERA5 has them) have no horizontal derivatives: the metric
        # cos(lat) is floored there and the horizontal diagnostics are zeroed.
        pole = np.abs(lat) > math.pi / 2 - 1e-6
        coslat = np.maximum(np.cos(lat), 0.5 * np.sin(np.diff(lat).min()))
        horizontal = (~pole).astype(np.float32)
        geo = horizontal * (np.abs(np.rad2deg(lat)) >= cfg.equator_mask_deg)
        self.register_buffer("lat", torch.tensor(lat, dtype=torch.float32), persistent=False)
        self.register_buffer("coslat", torch.tensor(coslat, dtype=torch.float32), persistent=False)
        self.register_buffer("coriolis", torch.tensor(2 * EARTH_OMEGA * np.sin(lat), dtype=torch.float32),
                             persistent=False)
        self.register_buffer("horizontal_mask", torch.tensor(horizontal), persistent=False)
        self.register_buffer("geo_mask", torch.tensor(geo, dtype=torch.float32), persistent=False)
        # Normalisation of the dynamic inputs; filled by `set_stats` and saved with the model.
        c = len(layout.dynamic)
        self.register_buffer("mean", torch.zeros(c))
        self.register_buffer("std", torch.ones(c))
        self.channels, self._keys = self._channel_list()

    # ----------------------------------------------------------- channels
    def _channel_list(self) -> Tuple[List[Channel], List[Tuple[str, Optional[int]]]]:
        fam = self.cfg.families
        keys: List[Tuple[str, Optional[int]]] = []
        if "hydrostatic" in fam:
            keys += [("hydrostatic", lo) for lo, _ in self.pairs]
        if "stability" in fam:
            keys += [("stability", lo) for lo, _ in self.pairs]
        if "geostrophic" in fam:
            keys += [("geostrophic", l) for l in self.levels]
            if self.surface_wind and self.surface_msl:
                keys.append(("geostrophic", None))
        if "saturation" in fam and self.humid:
            keys += [("saturation", l) for l in self.levels]
        if "kinematics" in fam:
            for name in ("divergence", "vorticity"):
                keys += [(name, l) for l in self.levels]
                if self.surface_wind:
                    keys.append((name, None))
        if "surface" in fam and self.surface_t2m:
            keys.append(("surface", None))
        return [Channel(f"diag_{n}", l) for n, l in keys], keys

    def __len__(self) -> int:
        return len(self.channels)

    def set_stats(self, stats) -> None:
        mean, std = stats.arrays(self.layout.dynamic)
        self.mean.copy_(torch.as_tensor(mean))
        self.std.copy_(torch.as_tensor(std))

    # ------------------------------------------------------ finite differences
    def _ddlon(self, x: Tensor) -> Tensor:
        return (x.roll(-1, -1) - x.roll(1, -1)) / (2 * self.dlon)

    def _ddlat(self, x: Tensor) -> Tensor:
        return torch.gradient(x, spacing=(self.lat,), dim=-2)[0]

    def _grad(self, phi: Tensor) -> Tuple[Tensor, Tensor]:
        c = self.coslat[:, None]
        return self._ddlon(phi) / (EARTH_RADIUS * c), self._ddlat(phi) / EARTH_RADIUS

    def _div_vort(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        c, m = self.coslat[:, None], self.horizontal_mask[:, None]
        div = (self._ddlon(u) + self._ddlat(v * c)) / (EARTH_RADIUS * c)
        vort = (self._ddlon(v) - self._ddlat(u * c)) / (EARTH_RADIUS * c)
        return div * m, vort * m

    def _geo_ratio(self, u: Tensor, v: Tensor, gx: Tensor, gy: Tensor) -> Tensor:
        f = self.coriolis[:, None]
        rx, ry = -f * v + gx, f * u + gy
        num = torch.sqrt(rx * rx + ry * ry + 1e-20)
        den = f.abs() * torch.sqrt(u * u + v * v + 1e-12) + torch.sqrt(gx * gx + gy * gy + 1e-20)
        return num / (den + 1e-9) * self.geo_mask[:, None]

    # ------------------------------------------------------------ diagnostics
    def physical(self, x: Tensor) -> Tensor:
        """Normalised dynamic inputs [B, C_dyn, H, W] -> physical units."""
        return x * self.std[:, None, None] + self.mean[:, None, None]

    def field(self, x: Tensor, name: str, level: Optional[int] = None) -> Tensor:
        return x[:, self._index[Channel(name, level)]]

    def raw(self, x: Tensor) -> Dict[Tuple[str, Optional[int]], Tensor]:
        """Physical dynamic fields [B, C_dyn, H, W] -> {(name, level): [B, H, W]}."""
        fam = self.cfg.families
        get = lambda n, l=None: self.field(x, n, l)
        out: Dict[Tuple[str, Optional[int]], Tensor] = {}
        t = {l: get("temperature", l).clamp_min(100.) for l in self.levels}  # a wild prediction stays finite
        q = {l: get("specific_humidity", l).clamp_min(0.) for l in self.levels} if self.humid else {}
        tv = {l: t[l] * (1 + 0.608 * q[l]) if q else t[l] for l in self.levels}
        z = {l: get("geopotential", l) for l in self.levels}
        for lo, up in self.pairs:
            thick = R_D * 0.5 * (tv[lo] + tv[up]) * math.log(lo / up)  # hydrostatic dPhi > 0
            if "hydrostatic" in fam:
                out[("hydrostatic", lo)] = (z[up] - z[lo]) / thick - 1
            if "stability" in fam:
                th_lo = t[lo] * (P_REF / (100. * lo)) ** KAPPA
                th_up = t[up] * (P_REF / (100. * up)) ** KAPPA
                out[("stability", lo)] = GRAVITY ** 2 * (th_up - th_lo) / (0.5 * (th_up + th_lo) * thick)
        winds = {l: (get("u_component_of_wind", l), get("v_component_of_wind", l)) for l in self.levels}
        if "geostrophic" in fam:
            for l in self.levels:
                out[("geostrophic", l)] = self._geo_ratio(*winds[l], *self._grad(z[l]))
            if ("geostrophic", None) in self._keys:
                gx, gy = self._grad(get("mean_sea_level_pressure") / RHO_SURFACE)
                out[("geostrophic", None)] = self._geo_ratio(
                    get("10m_u_component_of_wind"), get("10m_v_component_of_wind"), gx, gy)
        if "saturation" in fam and self.humid:
            for l in self.levels:
                p = 100. * l
                es = 611.2 * torch.exp(17.67 * (t[l] - 273.15) / (t[l] - 29.65).clamp_min(1.0))
                es = torch.minimum(es, torch.full_like(es, 0.5 * p))
                qs = 0.622 * es / (p - 0.378 * es)
                out[("saturation", l)] = 1 - q[l] / qs
        if "kinematics" in fam:
            for l in self.levels:
                out[("divergence", l)], out[("vorticity", l)] = self._div_vort(*winds[l])
            if self.surface_wind:
                out[("divergence", None)], out[("vorticity", None)] = self._div_vort(
                    get("10m_u_component_of_wind"), get("10m_v_component_of_wind"))
        if ("surface", None) in self._keys:
            out[("surface", None)] = get("2m_temperature") - t[1000]
        return out

    def forward(self, x: Tensor) -> Tensor:
        """Normalised dynamic inputs [B, C_dyn, H, W] -> squashed features [B, C_diag, H, W]."""
        with torch.autocast(x.device.type, enabled=False):  # differences of large numbers in fp32
            raw = self.raw(self.physical(x.float()))
            feats = [torch.tanh(raw[k] / SCALES[k[0]]) for k in self._keys]
        return torch.stack(feats, 1).to(x.dtype)
