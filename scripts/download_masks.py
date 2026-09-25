#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

GRIDS = {"coarse": "5.625", "fine": "1.5"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", choices=list(GRIDS), default="coarse")
    p.add_argument("--output-dir", default="data/era5")
    return p.parse_args()


def main():
    import cdsapi

    args = parse_args()
    grid = GRIDS[args.resolution]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"era5_masks_{grid}.nc"
    if target.exists():
        print(f"Already present: {target}")
        return

    part = target.with_suffix(".nc.part")
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels",
        {"product_type": "reanalysis", "format": "netcdf",
         "variable": ["land_sea_mask", "geopotential"],
         "grid": [grid, grid],
         "year": "2023", "month": "01", "day": "01", "time": "00:00"},
        str(part))
    os.replace(part, target)
    print(f"Downloaded: {target}")


if __name__ == "__main__":
    main()
