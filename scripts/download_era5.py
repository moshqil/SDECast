#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GRIDS = {"coarse": "5.625", "fine": "1.5"}
SINGLE_VARS = ["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"]
PRESSURE_VARS = ["geopotential", "temperature"]
PRESSURE_LEVELS = ["500", "850"]
ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_HOURS = [f"{t:02d}:00" for t in range(24)]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", choices=list(GRIDS), default="coarse",
                   help="coarse = 5.625 deg (32x64, what the released model uses)")
    p.add_argument("--years", type=int, nargs="+", default=[2018],
                   help="Years to fetch. The paper trains on 1979-2015, validates on "
                        "2016 and tests on 2018.")
    p.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)))
    p.add_argument("--output-dir", default="data/era5")
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent years. CDS throttles per user; 1-3 is sensible.")
    return p.parse_args()


def retrieve(client, dataset, request, target: Path) -> bool:
    if target.exists():
        return False
    part = target.with_suffix(target.suffix + ".part")
    client.retrieve(dataset, request, str(part))
    os.replace(part, target)
    return True


def download_year(year, grid, grid_label, months, out_dir: Path):
    import cdsapi
    client = cdsapi.Client()
    base = dict(product_type="reanalysis", format="netcdf", year=str(year),
                day=ALL_DAYS, time=ALL_HOURS, grid=[grid, grid])
    try:
        for m in months:
            month = f"{m:02d}"
            got = retrieve(client, "reanalysis-era5-single-levels",
                           {**base, "month": month, "variable": SINGLE_VARS},
                           out_dir / f"temp_single_{grid_label}_{year}_{month}.nc")
            got |= retrieve(client, "reanalysis-era5-pressure-levels",
                            {**base, "month": month, "variable": PRESSURE_VARS,
                             "pressure_level": PRESSURE_LEVELS},
                            out_dir / f"temp_pressure_{grid_label}_{year}_{month}.nc")
            print(f"[{year}-{month}] {'downloaded' if got else 'already present'}")
        print(f"[{year}] complete -- now run merge_era5.py for this year.")
    except Exception as exc:
        print(f"[{year}] FAILED: {exc}")
        raise


def main():
    args = parse_args()
    grid = GRIDS[args.resolution]
    grid_label = grid + "deg"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(args.years)} year(s) at {grid} deg into {out_dir}")
    if args.workers <= 1:
        for y in args.years:
            download_year(y, grid, grid_label, args.months, out_dir)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            list(ex.map(lambda y: download_year(y, grid, grid_label, args.months, out_dir),
                        args.years))
    print("Done. Remember to also fetch the static masks: scripts/download_masks.py")


if __name__ == "__main__":
    main()
