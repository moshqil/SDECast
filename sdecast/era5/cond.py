import os
import glob
import torch
import xarray as xr
import numpy as np

""""
Weighted RMSE and CRPS from https://github.com/Rose-STL-Lab/u-cast/
"""


def crop_pole_lat(arr, lat_axis=-2):
    if arr.shape[lat_axis] != 33:
        return arr
    idx = [slice(None)] * arr.ndim
    idx[lat_axis] = slice(0, 32)
    return arr[tuple(idx)]


def compute_area_weights(lat: np.ndarray) -> torch.Tensor:
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
    with xr.open_dataset(mask_file, engine='h5netcdf') as ds:
        orography_var = 'orography' if 'orography' in ds.variables else 'z'

        def _load_lat_lon(name):
            da = ds[name]
            extra = [d for d in da.dims if d not in ('latitude', 'longitude')]
            for d in extra:
                assert da.sizes[d] == 1, (
                    f"{name}: expected singleton non-spatial dim {d!r}, got size {da.sizes[d]}"
                )
            if extra:
                da = da.isel({d: 0 for d in extra})
            return da.transpose('latitude', 'longitude').values

        masks = np.stack(
            [_load_lat_lon('lsm'), _load_lat_lon(orography_var)], axis=0
        ).astype(np.float32)

        for c in range(masks.shape[0]):
            m, s = masks[c].mean(), masks[c].std()
            masks[c] = (masks[c] - m) / (s + 1e-8)

        masks = crop_pole_lat(masks)

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
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon_1d_deg), np.asarray(lat_1d_deg))
    lat_t = torch.tensor(lat_grid, dtype=torch.float32)
    lon_t = torch.tensor(lon_grid, dtype=torch.float32)
    return spatial_embeddings(lat_t, lon_t)


def temporal_embeddings_grid(t_days, H, W):
    B = t_days.shape[0]
    return temporal_embeddings(t_days).view(B, 4, 1, 1).expand(B, 4, H, W)


def build_static_era5_cond(spat_emb, masks=None):
    if masks is None:
        return spat_emb
    return torch.cat([spat_emb, masks], dim=0)


def assemble_era5_cond(time_emb, static_cond=None):
    if static_cond is None:
        return time_emb
    if static_cond.dim() == time_emb.dim() - 1:
        static_cond = static_cond.unsqueeze(0).expand(time_emb.shape[0], *static_cond.shape)
    if time_emb.shape[-2:] != static_cond.shape[-2:]:
        time_emb = time_emb.expand(*time_emb.shape[:-2], *static_cond.shape[-2:])
    return torch.cat([time_emb, static_cond], dim=1)