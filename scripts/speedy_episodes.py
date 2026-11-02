"""Stage 15: generate the SPEEDY episode dataset (simulator as teacher).

    python scripts/speedy_episodes.py --speedy SIM/speedy15 --nature DATA/speedy_wb2 --out DATA/speedy_episodes \
        --counts A=12 B=12 C=12 D=1 --days 30 --long-days 180 --test A=3 B=3 C=3 D=1

Needs the stage 15 SPEEDY build (scripts/setup_speedy.sh; per-process tendency
output and restart perturbations). Writes OUT/<name>.zarr per episode and
OUT/episodes.json (specs + split). Resumable: finished episodes are skipped.
--max-gb stops before the projected size exceeds the budget.
"""
import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path

import _path  # noqa: F401
from climate_wm.sim.episodes import dir_size, is_complete, plan_episodes, write_episode
from climate_wm.sim.speedy import SpeedyRunner


def _pairs(items):
    return {k: int(v) for k, v in (s.split("=") for s in items)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--speedy", required=True)
    ap.add_argument("--nature", required=True, help="nature-run root: restarts/, static.nc")
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts", nargs="+", default=["A=12", "B=12", "C=12", "D=1"])
    ap.add_argument("--test", nargs="+", default=["A=3", "B=3", "C=3", "D=1"])
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--long-days", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-gb", type=float, default=3.5)
    ap.add_argument("--only", nargs="*", default=None, help="episode names to write (default all)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dates = [dt.datetime.strptime(f.stem, "%Y%m%d") for f in sorted((Path(args.nature) / "restarts").glob("*.bin"))]
    specs = plan_episodes(dates, _pairs(args.counts), args.days, args.long_days, _pairs(args.test), args.seed)
    (out / "episodes.json").write_text(json.dumps([s.to_json() for s in specs], indent=1))
    runner = SpeedyRunner(args.speedy, str(out / "_work"), tendencies=True)
    total_days = sum(s.days for s in specs)
    for s in specs:
        if args.only and s.name not in args.only:
            continue
        tic = time.time()
        write_episode(runner, s, args.nature, str(out))
        # complete episodes only; the SPEEDY work directory (_work, raw netCDF of a killed run) is not stored data
        done = [e for e in specs if is_complete(out / f"{e.name}.zarr")]
        done_days = sum(e.days for e in done)
        size = sum(dir_size(out / f"{e.name}.zarr") for e in done) / 1e9
        projected = size / done_days * total_days
        print(f"[episodes] {s.name} ({s.group}, {s.split}, {s.start:%Y-%m-%d}, {s.days} d) "
              f"{time.time() - tic:.0f} s, {size:.2f} GB so far, projected {projected:.2f} GB", flush=True)
        if projected > args.max_gb and done_days < total_days:
            raise SystemExit(f"projected size {projected:.2f} GB > --max-gb {args.max_gb}")
    print(f"[episodes] done: {len(specs)} episodes in {out}")


if __name__ == "__main__":
    main()
