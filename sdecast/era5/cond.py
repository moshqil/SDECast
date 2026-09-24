import os
import glob
import torch
import xarray as xr
import numpy as np

""""
Weighted RMSE and CRPS from https://github.com/Rose-STL-Lab/u-cast/
"""

# ──────────────────────────────────────────────────────────────────────────
# TENSOR ORIENTATION CONVENTION (ERA5)
# Model-space tensors are (..., lat, lon): latitude on the 2nd-to-last axis (H),
# longitude on the last axis (W). Longitude is the periodic axis — SongUNet
# circular-pads the last (W) axis, so lon MUST stay last for the wrap to be
# physically correct. The dataloader crops latitude 33 -> 32 on the H axis.
#
# EXCEPTION: the area-weighted metric helpers (_area_weighted_*) follow
# U-Cast's (..., lon, lat) layout — lat is the LAST axis, where the cosine
# weights broadcast. Callers MUST transpose lat/lon before calling them
# (see evaluate.compute_per_channel_metrics).
# ──────────────────────────────────────────────────────────────────────────

def crop_pole_lat(arr, lat_axis=-2):
    """Drop the redundant pole row of the 5.625° grid (33 -> 32 latitudes).

    Args:
        arr: array/tensor with latitude on `lat_axis`.
        lat_axis: the latitude axis (default -2, the model-space H axis).
    """
    if arr.shape[lat_axis] != 33:
        return arr
    idx = [slice(None)] * arr.ndim
    idx[lat_axis] = slice(0, 32)
    return arr[tuple(idx)]


def compute_area_weights(lat: np.ndarray) -> torch.Tensor:
    """Compute cosine-latitude area weights, normalized to mean=1.

    Args:
        lat: 1-D latitudes in degrees, shape (lat,).
    Returns: (lat,) weights, to be broadcast over the latitude axis.
    """
    weights = np.cos(np.deg2rad(lat))
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def era5_file_path(data_path, year):
    matches = sorted(glob.glob(os.path.join(data_path, f"era5_1h_*_{year}.nc")))
    return matches[0] if matches else None


def era5_mask_path(data_path):
    matches = sorted(glob.glob(os.path.join(data_path, "era5_masks_*.nc")))
    return matches[0] if matches else None


def compute_and_save_normalization(data_path, years, out_file='era5_stats.pt', variables=['t2m', 'u10', 'v10', 'z500', 't850']):
    stats = {v: {'sum': 0.0, 'sum_sq': 0.0, 'count': 0} for v in variables}
    
    for year in years:
        file_path = era5_file_path(data_path, year)
        if file_path is None:
            continue

        with xr.open_dataset(file_path, engine='h5netcdf') as ds:
            for var in variables:
                arr = ds[var].values.astype(np.float64)
                stats[var]['sum'] += arr.sum()
                stats[var]['sum_sq'] += (arr ** 2).sum()
                stats[var]['count'] += arr.size
                
    means = []
    stds = []
    
    for var in variables:
        mean = stats[var]['sum'] / stats[var]['count']
        var_sq = (stats[var]['sum_sq'] / stats[var]['count']) - (mean ** 2)
        std = np.sqrt(max(var_sq, 0.0))
        means.append(mean)
        stds.append(std)
        
    mean_tensor = torch.tensor(means, dtype=torch.float32)
    std_tensor = torch.tensor(stds, dtype=torch.float32)

    torch.save({'mean': mean_tensor, 'std': std_tensor}, out_file)
    print(f"Saved ERA5 normalization stats to: {os.path.abspath(out_file)}")
    for var, m, s in zip(variables, means, stds):
        print(f"  {var}: mean={m:.6g}, std={s:.6g}")

def load_era5_masks(mask_file):
    """Load land-sea mask and geopotential from an ERA5 mask netcdf file.
    Returns a tensor of shape (C_mask, lat, lon) where C_mask=2 (lsm, z).
    """
    with xr.open_dataset(mask_file, engine='h5netcdf') as ds:
        orography_var = 'orography' if 'orography' in ds.variables else 'z'

        def _load_lat_lon(name):
            da = ds[name]
            # collapse any non-spatial dims (e.g. the singleton valid_time axis);
            # assert they're singletons so we never silently drop real data
            extra = [d for d in da.dims if d not in ('latitude', 'longitude')]
            for d in extra:
                assert da.sizes[d] == 1, (
                    f"{name}: expected singleton non-spatial dim {d!r}, got size {da.sizes[d]}"
                )
            if extra:
                da = da.isel({d: 0 for d in extra})
            # enforce (lat, lon) regardless of stored order: WB2 fine masks are
            # stored (lon, lat), coarse copernicus masks (lat, lon)
            return da.transpose('latitude', 'longitude').values

        masks = np.stack(
            [_load_lat_lon('lsm'), _load_lat_lon(orography_var)], axis=0
        ).astype(np.float32)

        for c in range(masks.shape[0]):
            m, s = masks[c].mean(), masks[c].std()
            masks[c] = (masks[c] - m) / (s + 1e-8)

        masks = crop_pole_lat(masks)  # (C, lat, lon): lat axis is -2

        return torch.tensor(masks, dtype=torch.float32)
    

def spatial_embeddings(lat, lon):
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)

    sin_lat, cos_lat = torch.sin(lat_rad), torch.cos(lat_rad)
    sin_lon, cos_lon = torch.sin(lon_rad), torch.cos(lon_rad)
    
    spherical_1 = sin_lat * cos_lon
    spherical_2 = sin_lat * sin_lon
    
    return torch.stack([sin_lat, cos_lat, sin_lon, cos_lon, spherical_1, spherical_2], dim=0)


def temporal_embeddings(t_days):
    sin_d = torch.sin(2 * torch.pi * t_days)
    cos_d = torch.cos(2 * torch.pi * t_days)

    sin_y = torch.sin(2 * torch.pi * t_days / 365.25)
    cos_y = torch.cos(2 * torch.pi * t_days / 365.25)
    
    return torch.stack([sin_d, cos_d, sin_y, cos_y], dim=1)


def spatial_embeddings_grid(lat_1d_deg, lon_1d_deg):
    """Build spatial embeddings on the lat/lon grid. Returns (6, lat, lon)."""
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon_1d_deg), np.asarray(lat_1d_deg))
    lat_t = torch.tensor(lat_grid, dtype=torch.float32)
    lon_t = torch.tensor(lon_grid, dtype=torch.float32)
    return spatial_embeddings(lat_t, lon_t)


def temporal_embeddings_grid(t_days, H, W):
    """Expand temporal embeddings across a spatial grid. Returns (B, 4, lat, lon)."""
    B = t_days.shape[0]
    return temporal_embeddings(t_days).view(B, 4, 1, 1).expand(B, 4, H, W)


def build_static_era5_cond(spat_emb, masks=None):
    """Concatenate spatial embedding with optional masks along the channel dim.
    All inputs/outputs are (C, lat, lon); returns (C_static, lat, lon).
    """
    if masks is None:
        return spat_emb
    return torch.cat([spat_emb, masks], dim=0)


def assemble_era5_cond(time_emb, static_cond=None):
    """Assemble full ERA5 cond tensor: [time_emb, static_cond] along channel dim.
    time_emb is (B, 4, lat, lon), static_cond (C_static, lat, lon); returns
    (B, C_cond, lat, lon).
    """
    if static_cond is None:
        return time_emb
    if static_cond.dim() == time_emb.dim() - 1:
        static_cond = static_cond.unsqueeze(0).expand(time_emb.shape[0], *static_cond.shape)
    # time_emb may carry size-1 spatial dims (broadcast per step during rollout);
    # expand to the static grid before concatenating.
    if time_emb.shape[-2:] != static_cond.shape[-2:]:
        time_emb = time_emb.expand(*time_emb.shape[:-2], *static_cond.shape[-2:])
    return torch.cat([time_emb, static_cond], dim=1)