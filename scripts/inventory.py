"""Stage 0 inventory of the ERA5 archive against a GraphCast task.

    python scripts/inventory.py --era5 configs/era5_source.yaml --task graphcast_13
"""

import argparse

import _path  # noqa: F401  (adds src/ to sys.path)
import yaml

from climate_wm.data import graphcast_mapping as gm
from climate_wm.data.era5_lazy_loader import open_era5
from climate_wm.data.inventory import format_report, inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5", required=True)
    parser.add_argument("--task", default="graphcast_13")
    args = parser.parse_args()
    with open(args.era5) as f:
        ds = open_era5(yaml.safe_load(f))
    print(format_report(inventory(ds, gm.get_task(args.task))))


if __name__ == "__main__":
    main()
