#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

GRID_LABELS = {"coarse": "5.625deg", "fine": "1.5deg"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resolution", choices=list(GRID_LABELS), default="coarse")
    p.add_argument("--years", type=int, nargs="+", default=[2018])
    p.add_argument("--output-dir", default="data/era5")
    p.add_argument("--keep-monthly", action="store_true",
                   help="Keep the temp_* monthly files after a successful merge")
    return p.parse_args()


def merge_year(year, grid_label, out_dir: Path, keep_monthly: bool):
    final = out_dir / f"era5_1h_{grid_label}_{year}.nc"
    if final.exists():
        print(f"[{year}] already merged: {final}")
        return

    single = [out_dir / f"temp_single_{grid_label}_{year}_{m:02d}.nc" for m in range(1, 13)]
    pressure = [out_dir / f"temp_pressure_{grid_label}_{year}_{m:02d}.nc" for m in range(1, 13)]
    missing = [p for p in single + pressure if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[{year}] {len(missing)} monthly file(s) missing, first: {missing[0]}. "
            "Run download_era5.py for this year first.")

    with xr.open_mfdataset(single, combine="nested", concat_dim="valid_time") as ds_s, \
         xr.open_mfdataset(pressure, combine="nested", concat_dim="valid_time") as ds_p:
        z500 = ds_p["z"].sel(pressure_level=500).drop_vars("pressure_level").rename("z500")
        t850 = ds_p["t"].sel(pressure_level=850).drop_vars("pressure_level").rename("t850")
        merged = xr.merge([ds_s, z500, t850])
        part = final.with_suffix(".nc.part")
        merged.to_netcdf(part)
    part.replace(final)
    print(f"[{year}] merged -> {final}")

    if not keep_monthly:
        for p in single + pressure:
            p.unlink()
        print(f"[{year}] removed 24 monthly temp files")


def main():
    args = parse_args()
    grid_label = GRID_LABELS[args.resolution]
    out_dir = Path(args.output_dir)
    for year in args.years:
        merge_year(year, grid_label, out_dir, args.keep_monthly)


if __name__ == "__main__":
    main()
