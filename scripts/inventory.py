"""Stage 0 inventory of the ERA5 archive against a GraphCast task.

    python scripts/inventory.py --era5 configs/era5_source.yaml --task graphcast_13
    python scripts/inventory.py --archive configs/era5_lustre_hourly.yaml --task graphcast_13_no_w

`--archive` scans the hourly archive with `HourlyArchive` (writes / extends its
index cache), lists every variable with its time coverage, and checks that each
GraphCast input, target, forcing and static field can be read, reading one frame.
"""

import argparse

import _path  # noqa: F401  (adds src/ to sys.path)
import yaml

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import open_era5
from climate_wm.data.inventory import format_report, inventory
from climate_wm.data.sources import HourlyArchive


def archive_report(path: str, task_name: str) -> None:
    import numpy as np
    with open(path) as f:
        archive = HourlyArchive(yaml.safe_load(f))
    print(archive.summary())
    task = gm.get_task(task_name)
    layout = gm.ChannelLayout(task, None)
    names = {c.name for c in layout.targets}
    missing = sorted(n for n in names | {c.name for c in layout.static} if not archive.has(n))
    if not archive.has("toa_incident_solar_radiation"):
        print("toa_incident_solar_radiation not in archive: it will be computed (forcings.py)")
    if missing:
        print(f"MISSING for {task_name}: {missing}")
        return
    ok = archive.availability(names)
    print(f"{ok.sum()} of {len(ok)} frames have every {task_name} target variable")
    first = int(np.argmax(ok))
    frame = archive.read(layout.targets, [first])
    static = archive.read_static(layout.static)
    print(f"read {archive.times[first]}: frame {frame.shape}, static {static.shape}, "
          f"NaN {int(np.isnan(frame).sum())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5", default=None)
    parser.add_argument("--archive", default=None, help="hourly archive config (HourlyArchive)")
    parser.add_argument("--task", default="graphcast_13")
    args = parser.parse_args()
    if args.archive:
        archive_report(args.archive, args.task)
        return
    with open(args.era5) as f:
        ds = open_era5(yaml.safe_load(f))
    print(format_report(inventory(ds, gm.get_task(args.task))))


if __name__ == "__main__":
    main()
