"""Downloads WeatherBench-2 subsets for a GraphCast-protocol experiment.

    python scripts/download_weatherbench2.py --out data/wb2_64x32 \
        --start 2016-01-01 --end 2019-01-15

Writes three local zarr stores (anonymous public GCS access, needs gcsfs):
  era5.zarr         ERA5 6-hourly, GraphCast task variables + static fields
  climatology.zarr  ERA5 1990-2017 hourly climatology (for ACC, eq. 29)
  graphcast.zarr    GraphCast forecasts for one year (WB2 has 2018 and 2020;
                    00/12 UTC inits, 6h..240h)
all on the same grid (default 64x32 equiangular, conservative regridding).
"""

import argparse
import os

import _path  # noqa: F401  (adds src/ to sys.path)
import xarray as xr

from climate_wm.data import graphcast_mapping as gm

WB2 = "gs://weatherbench2/datasets"
GRIDS = {
    "64x32": dict(
        era5="era5/1959-2022-6h-64x32_equiangular_conservative.zarr",
        clim="era5-hourly-climatology/1990-2017_6h_64x32_equiangular_conservative.zarr",
        graphcast="graphcast/{y}/date_range_{y0}-11-16_{y1}-02-01_12_hours-64x32_equiangular_conservative.zarr"),
    "240x121": dict(
        era5="era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
        clim="era5-hourly-climatology/1990-2017_6h_240x121_equiangular_with_poles_conservative.zarr",
        graphcast="graphcast/{y}/date_range_{y0}-11-16_{y1}-02-01_12_hours-240x121_equiangular_with_poles_conservative.zarr"),
}


def _open(path):
    return xr.open_zarr(f"{WB2}/{path}", storage_options={"token": "anon"})


def _write(ds, path, time_chunk=64):
    ds = ds.chunk({d: -1 for d in ds.dims if d != "time"})
    if "time" in ds.dims:
        ds = ds.chunk({"time": time_chunk})
    for v in ds.variables:
        ds[v].encoding = {}
    ds.to_zarr(path, mode="w")
    print(f"wrote {path}: {dict(ds.sizes)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--grid", default="64x32", choices=sorted(GRIDS))
    parser.add_argument("--task", default="graphcast_13")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2019-01-15")
    parser.add_argument("--forecast-year", type=int, default=2018, choices=[2018, 2020])
    parser.add_argument("--forecast-start", default=None, help="default: Jan 1 of the year")
    parser.add_argument("--forecast-end", default=None, help="default: Dec 31 12 UTC")
    parser.add_argument("--forecast-lead-steps", type=int, default=40, help="6h steps to keep")
    parser.add_argument("--forecast-init-stride", type=int, default=1, help="keep every n-th init")
    parser.add_argument("--skip", nargs="*", default=[], choices=["era5", "clim", "graphcast"])
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    task = gm.TASKS[args.task]
    levels = list(task.pressure_levels)
    targets = list(task.target_variables)
    paths = GRIDS[args.grid]

    if "era5" not in args.skip:
        era5 = _open(paths["era5"])
        dynamic = targets + ["toa_incident_solar_radiation"]
        ds = era5[dynamic].sel(time=slice(args.start, args.end), level=levels)
        ds = ds.merge(era5[list(gm.STATIC_VARS)])
        _write(ds, os.path.join(args.out, "era5.zarr"))
    if "clim" not in args.skip:
        clim = _open(paths["clim"])[targets].sel(level=levels)
        _write(clim, os.path.join(args.out, "climatology.zarr"))
    if "graphcast" not in args.skip:
        y = args.forecast_year
        gc = _open(paths["graphcast"].format(y=y, y0=y - 1, y1=y + 1))
        keep = [v for v in targets if v in gc]
        start = args.forecast_start or f"{y}-01-01"
        end = args.forecast_end or f"{y}-12-31T12:00"
        gc = gc[keep].sel(time=slice(start, end), level=levels)
        gc = gc.isel(time=slice(None, None, args.forecast_init_stride),
                     prediction_timedelta=slice(0, args.forecast_lead_steps))
        _write(gc, os.path.join(args.out, "graphcast.zarr"), time_chunk=1)  # one init per chunk


if __name__ == "__main__":
    main()
