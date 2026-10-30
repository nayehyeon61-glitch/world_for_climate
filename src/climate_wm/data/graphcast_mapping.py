"""GraphCast task specification, ERA5 name mapping and channel layout.

Variable names, pressure levels and task definitions are copied from
`weathernext/weathernext1_graph/graphcast.py` and
`weathernext/utils/variables.py` so that inputs / targets / forcings are the
same as GraphCast's `TaskConfig`:

    inputs  : X[t-6h], X[t]  (target vars [+ precip] + forcings + static)
    forcings: forcing vars at the target time t+6h
    targets : X[t+6h]        (5 surface + 6 x levels atmospheric)

Our ERA5 archive has no vertical velocity, so `graphcast_13_no_w` (plan B of
the research design) is the default; `graphcast_13` is the exact GraphCast
13-level task once `w` is available.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..geometry.shells import ShellSpec

PRESSURE_LEVELS_ERA5_37 = (
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300,
    350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900,
    925, 950, 975, 1000)
PRESSURE_LEVELS_WEATHERBENCH_13 = (
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)

TARGET_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "total_precipitation_6hr",
)
TARGET_SURFACE_NO_PRECIP_VARS = TARGET_SURFACE_VARS[:4]
TARGET_ATMOSPHERIC_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
TARGET_ATMOSPHERIC_NO_W_VARS = tuple(
    v for v in TARGET_ATMOSPHERIC_VARS if v != "vertical_velocity")
EXTERNAL_FORCING_VARS = ("toa_incident_solar_radiation",)
TIME_FORCING_VARS = (
    "year_progress_sin",
    "year_progress_cos",
    "day_progress_sin",
    "day_progress_cos",
)
FORCING_VARS = EXTERNAL_FORCING_VARS + TIME_FORCING_VARS
STATIC_VARS = ("geopotential_at_surface", "land_sea_mask")
ALL_ATMOSPHERIC_VARS = (
    "potential_vorticity", "specific_rain_water_content",
    "specific_snow_water_content", "geopotential", "temperature",
    "u_component_of_wind", "v_component_of_wind", "specific_humidity",
    "vertical_velocity", "vorticity", "divergence", "relative_humidity",
    "ozone_mass_mixing_ratio", "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content", "fraction_of_cloud_cover",
)

# Tangent vector fields on S^2 (u = eastward, v = northward component).
VECTOR_PAIRS = (
    ("u_component_of_wind", "v_component_of_wind"),
    ("10m_u_component_of_wind", "10m_v_component_of_wind"),
)

# ERA5 short names (our Lustre archive) -> GraphCast / WeatherBench names.
ERA5_SHORT_TO_GRAPHCAST = {
    "z": "geopotential",
    "q": "specific_humidity",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "mslp": "mean_sea_level_pressure",
    "t2m": "2m_temperature",
    "tp": "total_precipitation",
    "tisr": "toa_incident_solar_radiation",
    "lsm": "land_sea_mask",
    "orography": "geopotential_at_surface",
    "z_sfc": "geopotential_at_surface",
    # Present in our archive, not used by GraphCast (optional extra inputs).
    "tcwv": "total_column_water_vapour",
    "sp": "surface_pressure",
    "skt": "skin_temperature",
    "d2m": "2m_dewpoint_temperature",
    "swrd": "surface_solar_radiation_downwards",
    "lwrd": "surface_thermal_radiation_downwards",
}

# GraphCast per-variable loss weights (graphcast.GraphCast.loss_and_predictions).
GRAPHCAST_PER_VARIABLE_WEIGHTS = {
    "2m_temperature": 1.0,
    "10m_u_component_of_wind": 0.1,
    "10m_v_component_of_wind": 0.1,
    "mean_sea_level_pressure": 0.1,
    "total_precipitation_6hr": 0.1,
}


@dataclasses.dataclass(frozen=True)
class TaskConfig:
    """Same fields as weathernext.utils.task.TaskConfig (+ the time step)."""
    input_variables: Tuple[str, ...]
    target_variables: Tuple[str, ...]
    forcing_variables: Tuple[str, ...]
    pressure_levels: Tuple[int, ...]
    input_duration: str = "12h"
    timestep: str = "6h"

    @property
    def input_steps(self) -> int:
        # extract_input_target_times keeps times in (-input_duration, 0].
        return int(pd.Timedelta(self.input_duration) / pd.Timedelta(self.timestep))

    @property
    def dynamic_input_variables(self) -> Tuple[str, ...]:
        excluded = set(self.forcing_variables) | set(STATIC_VARS)
        return tuple(v for v in self.input_variables if v not in excluded)

    @property
    def static_variables(self) -> Tuple[str, ...]:
        return tuple(v for v in self.input_variables if v in STATIC_VARS)

    def with_levels(self, levels: Sequence[int]) -> "TaskConfig":
        return dataclasses.replace(self, pressure_levels=tuple(levels))


def _task(surface, atmospheric, levels, precip_in=True) -> TaskConfig:
    inputs_surface = surface if precip_in else tuple(
        v for v in surface if v != "total_precipitation_6hr")
    return TaskConfig(
        input_variables=inputs_surface + atmospheric + FORCING_VARS + STATIC_VARS,
        target_variables=surface + atmospheric,
        forcing_variables=FORCING_VARS,
        pressure_levels=levels,
    )


TASKS: Dict[str, TaskConfig] = {
    # graphcast.TASK (GraphCast paper model, 37 levels).
    "graphcast": _task(TARGET_SURFACE_VARS, TARGET_ATMOSPHERIC_VARS,
                       PRESSURE_LEVELS_ERA5_37),
    # graphcast.TASK_13 (GraphCast_small task definition).
    "graphcast_13": _task(TARGET_SURFACE_VARS, TARGET_ATMOSPHERIC_VARS,
                          PRESSURE_LEVELS_WEATHERBENCH_13),
    # graphcast.TASK_13_PRECIP_OUT (GraphCast_operational).
    "graphcast_operational": _task(TARGET_SURFACE_VARS, TARGET_ATMOSPHERIC_VARS,
                                   PRESSURE_LEVELS_WEATHERBENCH_13,
                                   precip_in=False),
    # Plan B: no vertical velocity in our ERA5 archive.
    "graphcast_13_no_w": _task(TARGET_SURFACE_VARS, TARGET_ATMOSPHERIC_NO_W_VARS,
                               PRESSURE_LEVELS_WEATHERBENCH_13),
    "graphcast_operational_no_w": _task(
        TARGET_SURFACE_VARS, TARGET_ATMOSPHERIC_NO_W_VARS,
        PRESSURE_LEVELS_WEATHERBENCH_13, precip_in=False),
}


def get_task(name: str, pressure_levels: Optional[Sequence[int]] = None
             ) -> TaskConfig:
    task = TASKS[name]
    return task.with_levels(pressure_levels) if pressure_levels else task


@dataclasses.dataclass(frozen=True)
class Channel:
    name: str
    level: Optional[int] = None

    @property
    def key(self) -> str:
        return self.name if self.level is None else f"{self.name}@{self.level}"


@dataclasses.dataclass
class ShellChannels:
    """Indices (into a flat channel list) that belong to one shell."""
    scalar: List[int]
    vector_u: List[int]
    vector_v: List[int]


def stack_channels(variables: Sequence[str], levels: Sequence[int]) -> List[Channel]:
    """GraphCast `dataset_to_stacked` order: variable-major, then level."""
    out = []
    for var in variables:
        if var in ALL_ATMOSPHERIC_VARS:
            out.extend(Channel(var, int(l)) for l in levels)
        else:
            out.append(Channel(var))
    return out


class ChannelLayout:
    """Flat channel ordering of the grid tensors and their shell partition."""

    def __init__(self, task: TaskConfig, shells: Optional[ShellSpec] = None):
        self.task = task
        self.shells = shells or ShellSpec.from_levels(task.pressure_levels)
        levels = task.pressure_levels
        self.dynamic = stack_channels(task.dynamic_input_variables, levels)
        self.targets = stack_channels(task.target_variables, levels)
        self.forcings = stack_channels(task.forcing_variables, levels)
        self.static = stack_channels(task.static_variables, levels)
        target_index = {c: i for i, c in enumerate(self.targets)}
        missing = [c.key for c in self.dynamic if c not in target_index]
        if missing:
            raise ValueError(f"Dynamic inputs must be targets too: {missing}")
        # Encoder input for any time with a target frame = targets[target_to_dynamic].
        self.target_to_dynamic = [target_index[c] for c in self.dynamic]
        self.dynamic_shells = self.partition(self.dynamic)
        self.target_shells = self.partition(self.targets)

    @property
    def num_encoder_channels(self) -> int:
        return len(self.dynamic) + len(self.forcings) + len(self.static)

    def shell_of(self, channel: Channel) -> int:
        if channel.level is None:
            return self.shells.surface_index
        return self.shells.shell_of_level(channel.level)

    def partition(self, channels: Sequence[Channel]) -> List[ShellChannels]:
        index = {c: i for i, c in enumerate(channels)}
        parts = [ShellChannels([], [], []) for _ in range(len(self.shells))]
        paired = set()
        for u_name, v_name in VECTOR_PAIRS:
            for c in channels:
                if c.name != u_name:
                    continue
                partner = Channel(v_name, c.level)
                if partner in index:
                    part = parts[self.shell_of(c)]
                    part.vector_u.append(index[c])
                    part.vector_v.append(index[partner])
                    paired.update([c, partner])
        for i, c in enumerate(channels):
            if c not in paired:
                parts[self.shell_of(c)].scalar.append(i)
        return parts

    def keys(self, channels: Sequence[Channel]) -> List[str]:
        return [c.key for c in channels]

    def summary(self) -> str:
        lines = [f"task levels={list(self.task.pressure_levels)} "
                 f"dynamic={len(self.dynamic)} targets={len(self.targets)} "
                 f"forcings={len(self.forcings)} static={len(self.static)}"]
        for shell, dyn, tgt in zip(self.shells.shells, self.dynamic_shells,
                                   self.target_shells):
            lines.append(
                f"  shell {shell.name:<18} levels={list(shell.levels)} "
                f"in(scalar={len(dyn.scalar)}, vector={len(dyn.vector_u)}) "
                f"out(scalar={len(tgt.scalar)}, vector={len(tgt.vector_u)})")
        return "\n".join(lines)
