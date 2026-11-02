"""Stage 0: inventory of an ERA5 archive against a GraphCast task.

Reports, for the standardized dataset, which task variables / levels are
present, the time coverage and spacing, units, and a sampled missing-value
rate, and flags the known gaps of our archive (e.g. no vertical velocity ->
use the `graphcast_13_no_w` task, plan B).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import xarray as xr

from .graphcast_mapping import STATIC_VARS, TaskConfig


def inventory(ds: xr.Dataset, task: TaskConfig, sample_times: int = 4) -> Dict:
    report: Dict = {"variables": {}, "missing_variables": [], "missing_levels": {},
                    "notes": []}
    times = ds.time.values if "time" in ds.coords else np.array([])
    if len(times):
        steps = np.unique(np.diff(times)) if len(times) > 1 else []
        report["time"] = {"start": str(times[0]), "end": str(times[-1]),
                          "count": int(len(times)),
                          "steps": [str(s.astype("timedelta64[h]")) for s in steps]}
    sample = np.linspace(0, max(len(times) - 1, 0), num=min(sample_times, max(len(times), 1))).astype(int)
    wanted: List[str] = list(dict.fromkeys(
        task.input_variables + task.target_variables))
    derived = {"year_progress_sin", "year_progress_cos", "day_progress_sin",
               "day_progress_cos"}
    for var in wanted:
        if var in derived:
            continue
        if var not in ds:
            if var == "toa_incident_solar_radiation":
                report["notes"].append("tisr missing: it will be computed (forcings.py).")
                continue
            report["missing_variables"].append(var)
            continue
        da = ds[var]
        info = {"dims": list(da.dims), "units": da.attrs.get("units", "?")}
        if "level" in da.dims:
            have = set(int(l) for l in da.level.values)
            missing = [l for l in task.pressure_levels if l not in have]
            if missing:
                report["missing_levels"][var] = missing
        if "time" in da.dims and var not in STATIC_VARS and len(times):
            vals = da.isel(time=sample)
            if "level" in da.dims:
                vals = vals.sel(level=[l for l in task.pressure_levels
                                       if l in set(int(x) for x in da.level.values)])
            info["missing_rate"] = float(np.isnan(np.asarray(vals.values)).mean())
        report["variables"][var] = info
    if "vertical_velocity" in report["missing_variables"]:
        report["notes"].append(
            "vertical_velocity (w) is missing: the exact GraphCast 13-level task "
            "cannot be reproduced; use task 'graphcast_13_no_w' (plan B) and do "
            "not claim identical I/O with pretrained GraphCast.")
    for var in STATIC_VARS:
        if var in report["missing_variables"]:
            report["notes"].append(f"static field {var} missing: obtain ERA5 invariants.")
    return report


def format_report(report: Dict) -> str:
    lines = []
    if "time" in report:
        t = report["time"]
        lines.append(f"time: {t['start']} .. {t['end']} ({t['count']} steps, spacing {t['steps']})")
    for var, info in report["variables"].items():
        miss = info.get("missing_rate")
        miss = "" if miss is None else f" missing={miss:.3%}"
        lines.append(f"  [ok] {var:<36} units={info['units']}{miss}")
    for var in report["missing_variables"]:
        lines.append(f"  [--] {var:<36} NOT FOUND")
    for var, levels in report["missing_levels"].items():
        lines.append(f"  [!!] {var} missing levels {levels}")
    lines += [f"note: {n}" for n in report["notes"]]
    return "\n".join(lines)
