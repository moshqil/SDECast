# Sample data

Small slices committed so the quickstart runs with no download. They are **not**
the evaluation sets -- reproducing published numbers needs the full data (see the
main README).

| File | Size | Contents |
|---|---|---|
| `era5_1h_5.625deg_2018.nc` | 5.9 MB | First 8 days of 2018 (192 hourly frames), 5 variables, 33 x 64 |
| `era5_masks_5.625.nc` | 43 KB | Land-sea mask + surface geopotential (time-invariant) |
| `sqg_N64_1hrly_steps_48_sample.npy` | 1.6 MB | One SQG trajectory, 49 hourly frames, shape (49, 2, 64, 64) |

The ERA5 file keeps the `era5_1h_<grid>_<year>.nc` naming the loader globs for, so
a real downloaded year is a drop-in replacement. The pole row is dropped at load
time (33 -> 32 latitudes).

ERA5 data is redistributed under the Copernicus licence -- see `NOTICE`.
