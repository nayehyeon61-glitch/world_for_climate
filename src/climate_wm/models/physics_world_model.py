"""Stage 9 world model: primitive-equation core + learned column physics.

The state is physical (spectral u/v via vorticity-divergence, T, q, Phi_b on
the 13 data levels). Every `call_minutes` the column network reads one grid
column of the Gaussian grid and returns the tendencies the core cannot
compute: heating Q, moistening S, momentum forcing F and a lowest-level
geopotential tendency P_b. With `budget` on, precipitation is the column
moisture sink (E - int S dp/g, E >= 0 from the network) and the momentum
forcing has zero column mean apart from a surface drag. The surface head
diagnoses the single-level targets (2 m temperature, mean sea-level pressure,
10 m winds, precipitation without budget) from the lowest levels.

`carry_residual` adds the initial-time difference between the observed and
the diagnosed fields (scales beyond the truncation, the hydrostatic residual,
surface offsets) to every lead, so the core is not charged for what it cannot
represent; vertical velocity and precipitation are not carried. Arm N0 = core
only (no networks).

Stage 10 adds an optional adaptive memory kernel (models/memory_kernel.py):
a recurrent state m on the Gaussian grid, updated at every column call and
read into the column tendencies, carried in the state as "mem" (the core
passes it through); `posterior` re-observes with it.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..data import forcings as forcing_lib
from ..data.graphcast_mapping import ChannelLayout
from .memory_kernel import MemoryConfig, MemoryKernel
from .primitive_core import EPS_V, R_DRY, CoreConfig, PrimitiveCore

GRAV = 9.80665
LEVEL_VARS = {"temp": "temperature", "z": "geopotential", "u": "u_component_of_wind",
              "v": "v_component_of_wind", "omega": "vertical_velocity",
              "q": "specific_humidity"}
SURFACE_VARS = ("2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
                "10m_v_component_of_wind", "total_precipitation_6hr")
COLUMN_IN = ("temp", "q", "u", "v", "omega")
DAY = 86400.0
# cause analysis (eval/physics_causes.py): column-physics outputs per knock-out,
# and the inputs a path cut can hold at a reference value (COLUMN_IN, the
# lowest-level geopotential, the land-sea mask, the solar radiation)
COLUMN_KNOCKS = {"heating": ("Q",), "moistening": ("S",), "momentum": ("F_u", "F_v"),
                 "surface_pressure": ("P_b",)}
PATH_INPUTS = COLUMN_IN + ("phib", "land", "sun")
PATH_OUTPUTS = ("heating", "moistening", "momentum", "surface_pressure", "evaporation")

# output scales per day: heating K, moistening kg/kg, momentum m/s, Phi_b m^2/s^2,
# evaporation kg/m^2 (= mm)
SCALES = {"Q": 10.0, "S": 2e-3, "F": 10.0, "P_b": 500.0, "E": 5.0}


@dataclasses.dataclass
class ColumnConfig:
    enabled: bool = True
    hidden: int = 256
    layers: int = 3
    call_minutes: float = 60.0
    budget: bool = True
    use_forcings: bool = True

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ColumnConfig":
        return cls(**(d or {}))


@dataclasses.dataclass
class PhysicsModelConfig:
    core: CoreConfig = dataclasses.field(default_factory=CoreConfig)
    column: ColumnConfig = dataclasses.field(default_factory=ColumnConfig)
    carry_residual: bool = True
    grad_checkpoint: bool = True
    memory: Optional[MemoryConfig] = None  # stage 10

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PhysicsModelConfig":
        d = dict(d or {})
        d.pop("type", None)
        mem = d.pop("memory", None)
        return cls(core=CoreConfig.from_dict(d.pop("core", None)),
                   column=ColumnConfig.from_dict(d.pop("column", None)),
                   memory=MemoryConfig.from_dict(mem) if mem else None, **d)


def _mlp(d_in: int, hidden: int, d_out: int, layers: int) -> nn.Sequential:
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    last = nn.Linear(d, d_out)
    nn.init.zeros_(last.weight)  # start from the core alone (arm N0)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)


class PhysicsWorldModel(nn.Module):
    """Core + column physics with the ClimateWorldModel rollout interface."""

    def __init__(self, cfg: PhysicsModelConfig, layout: ChannelLayout,
                 lat: Sequence[float], lon: Sequence[float]):
        super().__init__()
        self.cfg, self.layout = cfg, layout
        self.levels = [int(l) for l in layout.task.pressure_levels]
        if self.levels != sorted(self.levels):
            raise ValueError("pressure levels must be listed top to bottom")
        self.lat = np.asarray(lat, np.float64)
        self.lon = np.asarray(lon, np.float64)
        self.tgt = self._indices(layout.targets)
        self.dyn = self._indices(layout.dynamic)
        self.static_keys = [c.key for c in layout.static]
        self.forcing_keys = [c.key for c in layout.forcings]
        self._build_core(self._standard_profile())
        n_lev = len(self.levels)
        self.n_low = min(3, n_lev)
        cc = cfg.column
        self.column = self.surface = None
        if cc.enabled:
            n_f = 5 if cc.use_forcings else 0
            d_in = len(COLUMN_IN) * n_lev + 1 + 2 + n_f + 1
            n_out = 4 * n_lev + 1 + (2 if cc.budget else 0)
            self.column = _mlp(d_in, cc.hidden, n_out, cc.layers)
            if cc.budget:  # drag starts near zero; evaporation at ~3.5 mm/day
                with torch.no_grad():
                    self.column[-1].bias[4 * n_lev + 2] = -3.0
            # surface head: lowest (3) levels of T, q, u, v + Phi_b + carried surface
            # values + static + forcings + sin(lat)
            s_in = 4 * self.n_low + 1 + 4 + 2 + n_f + 1
            self.surface = _mlp(s_in, cc.hidden // 2, 5, 2)
        self.memory = None
        if cfg.memory is not None and cfg.memory.channels > 0:
            if self.column is None:
                raise ValueError("the memory kernel reads and feeds the column physics")
            self.memory = MemoryKernel(cfg.memory, d_in, n_out, len(layout.dynamic), self.core.sh,
                                       cc.call_minutes * 60.0)
        self.register_buffer("mean", torch.zeros(len(layout.targets)))
        self.register_buffer("std", torch.ones(len(layout.targets)))
        self.register_buffer("in_mean", torch.zeros(len(layout.dynamic)))
        self.register_buffer("in_std", torch.ones(len(layout.dynamic)))
        self.register_buffer("static_mean", torch.zeros(len(layout.static)))
        self.register_buffer("static_std", torch.ones(len(layout.static)))
        self.register_buffer("forcing_mean", torch.zeros(len(layout.forcings)))
        self.register_buffer("forcing_std", torch.ones(len(layout.forcings)))
        # interventions for the cause analysis (eval/physics_causes.py), not saved
        self.column_knock: set = set()               # COLUMN_KNOCKS outputs set to zero
        self.forcing_override: Dict[str, float] = {}  # forcing key -> physical value (e.g. TISR := 0)
        # path cuts: PATH_OUTPUTS group (or "all") -> PATH_INPUTS it sees at their reference
        # (the global-mean profile; no land; no sun) while the other outputs see the state
        self.column_paths: Dict[str, Tuple[str, ...]] = {}

    # ------------------------------------------------------------ set-up
    def _indices(self, channels) -> Dict[str, object]:
        keys = [c.key for c in channels]
        pos = lambda k: keys.index(k) if k in keys else -1
        out = {name: [pos(f"{var}@{l}") for l in self.levels] for name, var in LEVEL_VARS.items()}
        for name in ("temp", "z", "u", "v", "q"):
            if -1 in out[name]:
                raise ValueError(f"the physics model needs {LEVEL_VARS[name]} on every level")
        if -1 in out["omega"]:
            out["omega"] = None  # optional target (no vertical velocity in the task)
        out["surface"] = [keys.index(k) if k in keys else -1 for k in SURFACE_VARS]
        return out

    def _standard_profile(self) -> np.ndarray:
        p = np.asarray(self.levels, np.float64)
        return np.where(p < 200, 216.0, 288.0 * (p / 1013.0) ** 0.19)  # rough mid-latitude

    def _build_core(self, t_ref) -> None:
        self.core = PrimitiveCore(self.cfg.core, self.levels, t_ref, self.lat, len(self.lon))
        sh = self.core.sh
        if getattr(self, "memory", None) is not None:
            self.memory.sh_ref[0] = sh  # the rebuilt core's transforms
        lat_g = sh.lat.double().numpy()
        self.lat_g = lat_g
        self.lon_g = np.arange(sh.nlon) * 360.0 / sh.nlon
        self.register_buffer("sinlat_g", torch.tensor(np.sin(np.deg2rad(lat_g)), dtype=torch.float32),
                             persistent=False)
        self.register_buffer("sinlat_d", torch.tensor(np.sin(np.deg2rad(self.lat)), dtype=torch.float32),
                             persistent=False)
        dp = np.diff(np.concatenate([[0.0], 0.5 * (np.asarray(self.levels[1:]) + self.levels[:-1]),
                                     [self.levels[-1] + 0.5 * (self.levels[-1] - self.levels[-2])]])) * 100.0
        self.register_buffer("dp_layer", torch.tensor(dp, dtype=torch.float32), persistent=False)

    def set_normalization(self, stats) -> None:
        ch = lambda cs: [c.key for c in cs]
        get = lambda d, keys: torch.tensor([d[k] for k in keys], dtype=torch.float32)
        self.mean.copy_(get(stats.mean, ch(self.layout.targets)))
        self.std.copy_(get(stats.std, ch(self.layout.targets)))
        self.in_mean.copy_(get(stats.mean, ch(self.layout.dynamic)))
        self.in_std.copy_(get(stats.std, ch(self.layout.dynamic)))
        self.static_mean.copy_(get(stats.mean, self.static_keys))
        self.static_std.copy_(get(stats.std, self.static_keys))
        self.forcing_mean.copy_(get(stats.mean, self.forcing_keys))
        self.forcing_std.copy_(get(stats.std, self.forcing_keys))
        t_ref = [stats.mean[f"temperature@{l}"] for l in self.levels]
        device = self.mean.device
        self._build_core(np.asarray(t_ref, np.float64))  # semi-implicit reference = climatology
        self.core.to(device)

    # ------------------------------------------------------- diagnostics
    def _level_fields(self, x: Tensor, idx) -> Dict[str, Tensor]:
        return {name: x[:, idx[name]] for name in LEVEL_VARS if idx[name] is not None}

    def diagnose(self, state, precip: Optional[Tensor] = None) -> Tensor:
        """Physical target-layout fields [B, C, H, W] from a core state (data grid)."""
        g = self.core.grid_fields(state)
        b = g["temp"].shape[0]
        out = g["temp"].new_zeros((b, len(self.layout.targets)) + g["temp"].shape[-2:])
        for name in LEVEL_VARS:
            if self.tgt[name] is not None:
                out[:, self.tgt[name]] = g[name]
        tv_b = g["temp"][:, -1] * (1 + EPS_V * g["q"][:, -1])
        surf = [g["temp"][:, -1], 100.0 * self.levels[-1] * torch.exp(g["z"][:, -1] / (R_DRY * tv_b)),
                g["u"][:, -1], g["v"][:, -1],
                precip if precip is not None else torch.zeros_like(tv_b)]
        for j, k in enumerate(self.tgt["surface"]):
            if k >= 0:
                out[:, k] = surf[j]
        return out

    def _forcing_grid(self, times_s: np.ndarray, gauss: bool = True) -> Tensor:
        """Normalised forcings [T, 5, H, W] at the given times on the Gaussian (or data) grid."""
        lat, lon = (self.lat_g, self.lon_g) if gauss else (self.lat, self.lon)
        t = np.asarray(times_s).astype("datetime64[s]")
        f = {"toa_incident_solar_radiation": forcing_lib.toa_incident_solar_radiation(t, lat, lon)}
        f.update(forcing_lib.time_forcings(t, lon, len(lat)))
        for key, value in self.forcing_override.items():
            f[key] = np.full_like(f[key], value)
        arr = np.stack([f[k] for k in self.forcing_keys], 1)
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(self.mean.device)
        return (x - self.forcing_mean[:, None, None]) / self.forcing_std[:, None, None]

    def _to_gauss(self, x: Tensor) -> Tensor:
        sh = self.core.sh
        return sh.to_grid(sh.from_data(x))

    # ------------------------------------------------------ column physics
    def column_forcing(self, state, static_g: Tensor, forc_g: Optional[Tensor]) -> Dict[str, Tensor]:
        """Grid tendencies of the column physics; with the memory kernel also the
        updated memory ("mem")."""
        sh, core = self.core.sh, self.core
        u, v = sh.uv(state["zeta"], state["div"])
        grid = {"temp": sh.to_grid(state["temp"]), "q": sh.to_grid(state["q"]), "u": u, "v": v,
                "omega": sh.to_grid(core.omega_spec(state["div"]))}
        idx = self.tgt
        feats = [(grid[n] - self.mean[idx[n]][:, None, None]) / self.std[idx[n]][:, None, None]
                 if idx[n] is not None else grid[n] / 0.1  # omega without stats: ~0.1 Pa/s
                 for n in COLUMN_IN]
        zb = sh.to_grid(state["phib"])[:, None]
        k = idx["z"][-1]
        feats.append((zb - self.mean[k]) / self.std[k])
        feats.append(static_g)
        if forc_g is not None:
            feats.append(forc_g)
        b = zb.shape[0]
        feats.append(self.sinlat_g[None, None, :, None].expand(b, 1, *zb.shape[-2:]))
        x = torch.cat(feats, 1).permute(0, 2, 3, 1)  # [B, H, W, C]
        if "all" in self.column_paths:
            x = self._hold_inputs(x, self.column_paths["all"])
        y = self.column(x)
        mem = None
        if self.memory is not None and "mem" in state:
            mem = self.memory.update(state["mem"], x)
            y = y + self.memory.tendency(x, mem)
        for group, inputs in self.column_paths.items():
            if group != "all":
                held = self._hold_inputs(x, inputs)
                y_held = self.column(held)
                if mem is not None:
                    y_held = y_held + self.memory.tendency(held, mem)
                j = self._path_outputs(group)
                y = y.clone()
                y[..., j] = y_held[..., j]
        y = y.permute(0, 3, 1, 2)  # [B, C_out, H, W]
        n = len(self.levels)
        per_s = 1.0 / DAY
        out = {"Q": y[:, :n] * SCALES["Q"] * per_s, "S": y[:, n:2 * n] * SCALES["S"] * per_s,
               "F_u": y[:, 2 * n:3 * n] * SCALES["F"] * per_s,
               "F_v": y[:, 3 * n:4 * n] * SCALES["F"] * per_s,
               "P_b": y[:, 4 * n] * SCALES["P_b"] * per_s}
        if self.cfg.column.budget:
            w = (self.dp_layer / self.dp_layer.sum())[None, :, None, None]
            cd = torch.nn.functional.softplus(y[:, 4 * n + 2]) * 1e-6  # C_D / depth, 1/m
            speed = torch.sqrt(u[:, -1] ** 2 + v[:, -1] ** 2 + 1e-6)
            low = torch.zeros(n, device=y.device)
            low[-3:] = 1.0  # drag acts on the lowest three levels
            for key, wind in (("F_u", u), ("F_v", v)):  # zero column mean + surface drag
                f = out[key] - (out[key] * w).sum(1, keepdim=True)
                out[key] = f - (cd * speed * wind[:, -1])[:, None] * low[None, :, None, None]
            evap = torch.nn.functional.softplus(y[:, 4 * n + 1]) * SCALES["E"] / DAY  # kg/m^2/s
            sink = (out["S"] * self.dp_layer[None, :, None, None]).sum(1) / GRAV
            out["precip"] = torch.relu(evap - sink)  # kg/m^2/s
        for name in self.column_knock:  # after the budget: precipitation is still diagnosed
            for key in COLUMN_KNOCKS[name]:
                out[key] = torch.zeros_like(out[key])
        if mem is not None:
            out["mem"] = mem
        return out

    def _hold_inputs(self, x: Tensor, inputs: Sequence[str]) -> Tensor:
        """Column input [B, H, W, C] with the PATH_INPUTS `inputs` at their reference."""
        n = len(self.levels)
        x = x.clone()
        for name in inputs:
            if name in COLUMN_IN:
                i = COLUMN_IN.index(name)
                x[..., i * n:(i + 1) * n] = 0.0  # normalised: the global-mean profile
            elif name == "phib":
                x[..., 5 * n] = 0.0
            elif name == "land":
                j = self.static_keys.index("land_sea_mask")
                x[..., 5 * n + 1 + j] = -self.static_mean[j] / self.static_std[j]
            elif name == "sun":
                if not self.cfg.column.use_forcings:
                    raise ValueError("the column sees no forcings")
                j = self.forcing_keys.index("toa_incident_solar_radiation")
                x[..., 5 * n + 3 + j] = -self.forcing_mean[j] / self.forcing_std[j]
            else:
                raise ValueError(f"unknown path input {name!r} (known: {PATH_INPUTS})")
        return x

    def _path_outputs(self, group: str) -> list:
        """Column output channels of a PATH_OUTPUTS group."""
        n = len(self.levels)
        budget = self.cfg.column.budget
        groups = {"heating": list(range(n)), "moistening": list(range(n, 2 * n)),
                  "momentum": list(range(2 * n, 4 * n)) + ([4 * n + 2] if budget else []),
                  "surface_pressure": [4 * n], "evaporation": [4 * n + 1] if budget else []}
        if group not in groups:
            raise ValueError(f"unknown path output {group!r} (known: {PATH_OUTPUTS})")
        return groups[group]

    def surface_correction(self, phys: Tensor, static: Tensor, forc: Optional[Tensor]) -> Tensor:
        """Normalised corrections [B, 5, H, W] for the single-level targets from the
        predicted fields `phys` (lowest levels and carried surface values)."""
        diag = carried = phys
        idx = self.tgt
        z = lambda name, sl: ((diag[:, [idx[name][i] for i in sl]] - self.mean[[idx[name][i] for i in sl]][:, None, None])
                              / self.std[[idx[name][i] for i in sl]][:, None, None])
        low = list(range(-self.n_low, 0))
        feats = [z("temp", low), z("q", low), z("u", low), z("v", low), z("z", [-1])]
        sk = [k for k in idx["surface"][:4]]
        feats.append((carried[:, sk] - self.mean[sk][:, None, None]) / self.std[sk][:, None, None])
        feats.append(static)
        if forc is not None:
            feats.append(forc)
        b = diag.shape[0]
        feats.append(self.sinlat_d[None, None, :, None].expand(b, 1, *diag.shape[-2:]))
        x = torch.cat(feats, 1).permute(0, 2, 3, 1)
        return self.surface(x).permute(0, 3, 1, 2)

    # ------------------------------------------------------------ rollout
    def initial_state(self, batch: Dict[str, Tensor]):
        x = batch["inputs"][:, -1] * self.in_std[:, None, None] + self.in_mean[:, None, None]
        f = self._level_fields(x, self.dyn)
        state = self.core.state_from_grid(f["u"], f["v"], f["temp"], f["q"], f["z"])
        obs = x.new_zeros((x.shape[0], len(self.layout.targets)) + x.shape[-2:])
        for name in f:
            obs[:, self.tgt[name]] = f[name]
        for kt, kd in zip(self.tgt["surface"], self.dyn["surface"]):
            if kt >= 0 and kd >= 0:
                obs[:, kt] = x[:, kd]
        return state, obs

    def carried_residual(self, state, obs: Tensor) -> Tensor:
        """obs - diagnosis of the initial state (zero for omega and precipitation)."""
        if not self.cfg.carry_residual:
            return torch.zeros_like(obs)
        resid = obs - self.diagnose(state)
        tp = self.tgt["surface"][4]
        if tp >= 0:
            resid[:, tp] = 0.0  # precipitation is accumulated, never carried
        if self.tgt["omega"] is not None:
            resid[:, self.tgt["omega"]] = 0.0  # diagnosed omega: carrying adds noise (checked)
        return resid

    def begin(self, batch: Dict[str, Tensor]):
        """Initial state and the rollout context of a batch."""
        state, obs = self.initial_state(batch)
        static = ((batch["static"] - self.static_mean[:, None, None]) / self.static_std[:, None, None])
        ctx = {"resid": self.carried_residual(state, obs), "static": static,
               "static_g": self._to_gauss(static) if self.column is not None else None,
               "init": batch["init_time"].cpu().numpy().astype(np.int64),
               "forcings_tgt": batch.get("forcings_tgt")}
        if self.memory is not None:
            sh = self.core.sh
            state["mem"] = static.new_zeros(static.shape[0], self.memory.channels, sh.nlat, sh.nlon)
        return state, ctx

    def reobserve(self, ctx, inputs: Tensor):
        """Restarts from observed (normalised dynamic) fields [B, C_dyn, H, W]."""
        state, obs = self.initial_state({"inputs": inputs[:, None]})
        ctx["resid"] = self.carried_residual(state, obs)
        return state

    def posterior(self, state, ctx, forecast: Tensor, truth: Tensor, mask: Tensor):
        """Stage 10 re-observation (normalised dynamic fields [B, C_dyn, H, W],
        mask 1 = hidden): the observed points restart from the truth, the hidden
        ones from the forecast plus the memory's readout R(m); the memory
        absorbs the innovation. Without memory this is nudging with strength 1."""
        seen = 1 - mask
        hidden = forecast
        mem = None
        if self.memory is not None:
            g = self._to_gauss
            mem = self.memory.absorb(state["mem"], g(forecast), g(seen * (truth - forecast)), g(seen))
            r = self.memory.readout(mem)
            if r is not None:
                sh = self.core.sh
                hidden = forecast + sh.to_grid(sh.to_spec(r), data=True)
        fresh = self.reobserve(ctx, seen * truth + mask * hidden)
        if mem is not None:
            fresh["mem"] = mem
        return fresh

    def advance(self, state, ctx, k: int):
        """Integrates lead k -> k + 1 (6 h). Returns the state and the data-grid
        precipitation of the step (None without the water budget)."""
        core, cc = self.core, self.cfg.column
        nsub = int(round(360.0 / core.cfg.dt_minutes))
        call_every = max(1, int(round(cc.call_minutes / core.cfg.dt_minutes)))
        forc_g = None
        if self.column is not None and cc.use_forcings:
            calls = np.arange(0, nsub, call_every)
            times = ctx["init"][:, None] + 21600 * k + (calls * core.cfg.dt_minutes * 60)[None]
            forc_g = torch.stack([self._forcing_grid(t) for t in times])  # [B, n_calls, 5, H, W]
        keys = list(state)

        def six_hours(*flat):
            st = dict(zip(keys, flat))
            precip = 0.0
            forcing = None
            for j in range(nsub):
                if self.column is not None and j % call_every == 0:
                    fg = forc_g[:, j // call_every] if forc_g is not None else None
                    forcing = self.column_forcing(st, ctx["static_g"], fg)
                    if "mem" in forcing:
                        st["mem"] = forcing.pop("mem")
                    if "precip" in forcing:
                        precip = precip + forcing["precip"] * cc.call_minutes * 60.0 / 1000.0  # m
                st = core.step(st, forcing)
            p = precip if torch.is_tensor(precip) else torch.zeros_like(st["temp"].real[:, 0, :1, :1])
            return tuple(st[key] for key in keys) + (p,)

        flat = tuple(state[key] for key in keys)
        if self.cfg.grad_checkpoint and torch.is_grad_enabled():
            out = checkpoint(six_hours, *flat, use_reentrant=False)
        else:
            out = six_hours(*flat)
        precip = out[-1]
        precip_d = None
        if precip.shape[-1] > 1:
            precip_d = core.sh.to_grid(core.sh.to_spec(precip), data=True)
        return dict(zip(keys, out[:-1])), precip_d

    def predict(self, state, ctx, k: int, precip_d: Optional[Tensor] = None) -> Tensor:
        """Normalised target-layout forecast [B, C, H, W] of lead k + 1."""
        cc = self.cfg.column
        phys = self.diagnose(state, precip_d) + ctx["resid"]
        norm = (phys - self.mean[:, None, None]) / self.std[:, None, None]
        if self.surface is not None:
            fd = ctx["forcings_tgt"][:, k] if cc.use_forcings else None
            corr = self.surface_correction(phys, ctx["static"], fd)
            for j, c in enumerate(self.tgt["surface"]):
                if c >= 0 and (j < 4 or not cc.budget):
                    norm[:, c] = norm[:, c] + corr[:, j]
        return norm

    def forward_rollout(self, batch: Dict[str, Tensor], steps: int) -> Tensor:
        """Normalised predictions [B, L, C, H, W] (differentiable)."""
        state, ctx = self.begin(batch)
        preds = []
        for k in range(steps):
            state, precip = self.advance(state, ctx, k)
            preds.append(self.predict(state, ctx, k, precip))
        return torch.stack(preds, 1)

    @torch.no_grad()
    def rollout(self, batch: Dict[str, Tensor], steps: int, members: int = 1, **_) -> Tensor:
        """Normalised predictions [M, B, L, C, H, W] (deterministic: members are copies)."""
        pred = self.forward_rollout(batch, steps)
        return pred[None].expand(members, *pred.shape)
