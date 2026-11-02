#!/usr/bin/env python
"""Stage 14: SPEEDY nature run in the WB2 layout (truth of the simulator world).

Spin-up from rest for one year, then 5-day segments chained by restart files.
Each segment's output is mapped to the WB2 64x32 / 13-level variables and
appended to a zarr store; the restart file at every segment start is kept, so
experiments (observation games, interventions) can branch from any 5-day point.

    python scripts/speedy_nature_run.py --speedy $S/sim/speedy --out $S/speedy_wb2 \
        --spinup 1981-01-01 --start 1982-01-01 --end 1985-01-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from climate_wm.sim.speedy import (SpeedyRunner, read_native, read_static, to_dataset,  # noqa: E402
                                   to_wb2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speedy", required=True, help="patched speedy.f90 root (bin/speedy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--spinup", default="1981-01-01")
    ap.add_argument("--start", default="1982-01-01")
    ap.add_argument("--end", default="1985-01-01")
    ap.add_argument("--segment-days", type=int, default=5)
    ap.add_argument("--flush-segments", type=int, default=6, help="segments per zarr append")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "restarts").mkdir(parents=True, exist_ok=True)
    runner = SpeedyRunner(args.speedy, str(out / "work"))
    spin, start, end = (dt.datetime.fromisoformat(x) for x in (args.spinup, args.start, args.end))
    store = out / "nature.zarr"

    first = out / "restarts" / f"{start:%Y%m%d}.bin"
    if not first.exists():
        t0 = time.time()
        run = runner.run("spinup", spin, start, static=True)
        shutil.copy(run / "restart.bin", first)
        shutil.copy(run / "static.nc", out / "static.nc")
        for f in run.glob("[0-9]*.nc"):     # the spin-up's own output is not kept
            f.unlink()
        print(f"spin-up {spin:%Y-%m-%d} -> {start:%Y-%m-%d}: {time.time() - t0:.0f} s", flush=True)
    static = read_static(out / "static.nc")

    done_until = start
    if store.exists():
        import xarray as xr
        last = xr.open_zarr(store).time.values[-1]
        done_until = dt.datetime.utcfromtimestamp(last.astype("datetime64[s]").astype(int))
        print(f"resuming after {done_until}", flush=True)

    buf, times = [], []
    t = done_until
    step = dt.timedelta(days=args.segment_days)
    t0 = time.time()
    while t < end:
        t1 = min(t + step, end)
        restart = out / "restarts" / f"{t:%Y%m%d}.bin"
        run = runner.run("nature", t, t1, restart_in=str(restart))
        shutil.copy(run / "restart.bin", out / "restarts" / f"{t1:%Y%m%d}.bin")
        seg_times = runner.output_times(run, t)
        buf.append(to_wb2(read_native(run, seg_times), static))
        times += seg_times
        t = t1
        if len(buf) >= args.flush_segments or t >= end:
            arrays = {k: (np.concatenate([b[k] for b in buf]) if buf[0][k].ndim > 2 else buf[0][k])
                      for k in buf[0]}
            ds = to_dataset(arrays, times, attrs={"source": "SPEEDY speedy.f90 T30 nature run"})
            if store.exists():
                ds.drop_vars(["geopotential_at_surface", "land_sea_mask"]).to_zarr(
                    store, append_dim="time")
            else:
                ds.chunk({"time": 120}).to_zarr(store)
            print(f"{t:%Y-%m-%d} written ({len(times)} times, {time.time() - t0:.0f} s)", flush=True)
            buf, times = [], []
    print("nature run done", flush=True)


if __name__ == "__main__":
    main()
