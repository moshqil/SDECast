"""ERA5 loader for inference.

Reads the merged yearly NetCDF files written by ``scripts/merge_era5.py`` (or the
committed sample slice) and returns initial conditions, ground truth and the
model's conditioning tensor.

Two conventions matter and are easy to get wrong:

* The pole row is dropped (33 latitudes -> 32) so the grid is 32x64, and latitude
  stays on axis -2 with longitude last -- the backbone wraps circularly in
  longitude only, so transposing them would be physically wrong.
* Spatial embeddings are derived from the file's own lat/lon grid rather than a
  stored tensor, so no training-time artefact is needed.
"""
from __future__ import annotations

import os

from typing import Optional, Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from sdecast.era5.cond import (
    assemble_era5_cond,
    build_static_era5_cond,
    compute_area_weights,
    crop_pole_lat,
    era5_file_path,
    era5_mask_path,
    load_era5_masks,
    spatial_embeddings_grid,
    temporal_embeddings_grid,
)

VAR_NAMES = ["t2m", "u10", "v10", "z500", "t850"]
TIME_DIM = "valid_time"


class ERA5Dataset(Dataset):
    """Yearly ERA5 NetCDF files as (state, conditioning, time) trajectories."""

    def __init__(self, data_path, years: Sequence[int] | None = None,
                 variables: Sequence[str] = VAR_NAMES,
                 stats_file: Optional[str] = None,
                 use_masks: bool = True,
                 normalize: bool = True):
        self.data_path = str(data_path)
        self.variables = list(variables)
        self.normalize = normalize
        self.use_masks = use_masks

        self.files = self._discover(years)
        if not self.files:
            raise FileNotFoundError(
                f"No ERA5 NetCDF files found in {self.data_path}"
                + (f" for years {list(years)}" if years else "")
                + ". Download with scripts/download_era5.py, or point at sample_data/."
            )

        if normalize:
            if stats_file is None:
                raise ValueError(
                    "stats_file is required when normalize=True. The released ERA5 "
                    "checkpoint expects weights/era5_stats.pt -- see weights/README.md."
                )
            stats = torch.load(str(stats_file), weights_only=True)
            self.mean = stats["mean"].view(1, len(self.variables), 1, 1).numpy()
            self.std = stats["std"].view(1, len(self.variables), 1, 1).numpy()
        else:
            self.mean = self.std = None

        with xr.open_dataset(self.files[0], engine="h5netcdf") as ds:
            self.lat = crop_pole_lat(ds["latitude"].values, lat_axis=0).astype(np.float32)
            self.lon = ds["longitude"].values.astype(np.float32)
            self.n_times = int(ds.sizes[TIME_DIM])

        mask_file = era5_mask_path(self.data_path)
        if use_masks and mask_file is None:
            raise FileNotFoundError(
                f"use_masks=True but no era5_masks_*.nc in {self.data_path}. The released "
                "ERA5 checkpoint was trained with masks (cond_channels=12) and needs them."
            )
        masks = load_era5_masks(mask_file) if use_masks else None
        self.static_cond = build_static_era5_cond(
            spatial_embeddings_grid(self.lat, self.lon), masks)
        self.area_weights = compute_area_weights(self.lat)

    def _discover(self, years):
        if os.path.isfile(self.data_path):
            return [self.data_path]
        if years is None:
            import glob
            return sorted(glob.glob(os.path.join(self.data_path, "era5_1h_*.nc")))
        out = []
        for y in years:
            p = era5_file_path(self.data_path, y)
            if p is not None:
                out.append(p)
        return out

    @property
    def cond_channels(self) -> int:
        return self.static_cond.shape[0] + 4  # + temporal embeddings

    def n_starts(self, length: int) -> int:
        """How many trajectories of ``length`` frames fit across all files."""
        return max(0, self.n_times - length) * len(self.files)

    def get_trajectory(self, idx: int, length: int = 24):
        """Return ``(data, cond, t_days)`` for trajectory ``idx``.

        ``data`` is (T, C, lat, lon) normalised; ``cond`` is (T, C_cond, lat, lon);
        ``t_days`` is (T,) as ``dayofyear + hour/24``.
        """
        per_file = max(1, self.n_times - length)
        file_idx = min(idx // per_file, len(self.files) - 1)
        start = idx % per_file
        if start + length > self.n_times:
            raise IndexError(
                f"Trajectory {idx} of length {length} runs past the end of "
                f"{self.files[file_idx]} ({self.n_times} frames)."
            )

        with xr.open_dataset(self.files[file_idx], engine="h5netcdf") as ds:
            subset = ds.isel({TIME_DIM: slice(start, start + length)})
            # Select by name: variable order must match the checkpoint's channel order.
            arr = subset[self.variables].to_array().transpose(
                TIME_DIM, "variable", "latitude", "longitude").values
            if self.normalize:
                arr = (arr - self.mean) / self.std
            arr = crop_pole_lat(arr)
            data = torch.tensor(arr, dtype=torch.float32)

            times = subset[TIME_DIM].dt
            t_days = torch.from_numpy(
                times.dayofyear.values + times.hour.values / 24.0).float()

        H, W = data.shape[-2], data.shape[-1]
        cond = assemble_era5_cond(temporal_embeddings_grid(t_days, H, W), self.static_cond)
        return data, cond, t_days

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map a normalised state back to physical units (K, m/s, m^2/s^2)."""
        if not self.normalize:
            return x
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        return x * std + mean
