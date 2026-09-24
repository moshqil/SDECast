# Model weights

The two checkpoints are **not committed** (`.gitignore` excludes `*.ckpt`). Place
them here as:

| File | Size | What it is |
|---|---|---|
| `era5_sdecast.ckpt` | 87 MB | SDE-Cast trained on ERA5 at 5.625 deg, 1 h lead |
| `sqg_sdecast.ckpt`  | 83 MB | SDE-Cast trained on SQG, learnable interpolant, 6 h bridge |
| `era5_stats.pt`     | 1.4 KB | ERA5 normalisation stats (committed) |

Everything the model needs is inside the checkpoint's `hyper_parameters`; there is
no separate config file. Inspect one with:

```python
from sdecast import load_hparams
print(load_hparams("weights/era5_sdecast.ckpt"))
```

## What is in them

|  | ERA5 | SQG |
|---|---|---|
| grid | 32 x 64 lat-lon, circular in longitude only | 64 x 64, doubly periodic |
| state channels | 5 (`z500`, `t850`, `t2m`, `u10`, `v10`) | 2 (surfaces z=0, z=H) |
| conditioning | 12 channels: 4 temporal + 6 spatial + 2 static | none |
| volatility | `full_const` -- one value per variable and grid cell | `const` -- one scalar |
| interpolant | `new_fixed` (learnable) | `new_fixed` (learnable) |
| drift network | 3,556,741 params | 3,542,050 params |
| interpolant net | 3,563,946 params | 3,544,932 params |
| training | epoch 49, step 63350 | epoch 17, step 6876 |

Both were written by the same training module, so one loader handles both. The
`h_hours=3` recorded in the ERA5 checkpoint is inert -- it only feeds the analytic
SQG drift and the divergence regulariser, neither of which that model uses.

## Normalisation stats

`era5_stats.pt` holds per-variable `mean` and `std` over the training period, in
the channel order above. It is **required** for ERA5 inference: the model works in
standardised space, and the same numbers convert forecasts back to physical units.

Two slightly different stat sets exist in the original research code (one fitted to
the Copernicus files, one to a WeatherBench copy; they differ by up to 2%, and the
sign of the `u10` mean flips). The file shipped here is the Copernicus-derived one,
matching what `scripts/download_era5.py` produces. The difference turns out not to
matter much -- a 1 h RMSE A/B over 2018 gave:

| stats | t2m | u10 | v10 | z500 | t850 |
|---|---|---|---|---|---|
| Copernicus (shipped) | 0.467 | 0.452 | 0.462 | 26.57 | 0.349 |
| WeatherBench | 0.465 | 0.452 | 0.464 | 26.52 | 0.347 |

If you retrain or use a different reanalysis, recompute with
`sdecast.era5.cond.compute_and_save_normalization`.
