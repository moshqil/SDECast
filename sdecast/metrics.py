import torch
import numpy as np
import torch.nn as nn
import math
from torch import Tensor

def rmse(pred, target, ens_mean=False):
    if len(pred.shape) != len(target.shape):
        target=target.unsqueeze(1)
    if ens_mean:
        pred = pred.mean(dim=1, keepdim=True)
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(-4, -3, -2, -1)))

def mae(pred, target, ens_mean=False):
    if len(pred.shape) != len(target.shape):
        target=target.unsqueeze(1)
    if ens_mean:
        pred = pred.mean(dim=1, keepdim=True)
    return torch.mean(torch.abs(pred - target), dim=(-3, -2, -1))

def spread(pred, target):
    pred_mean = pred.mean(dim=1, keepdim=True)
    return torch.sqrt(torch.mean((pred - pred_mean) ** 2, dim=(-4, -3, -2, -1)))

def spread_skill(pred, target):
    if len(pred.shape) != len(target.shape):
        target=target.unsqueeze(1)
    skill = rmse(pred, target, ens_mean=True)
    spr = spread(pred, target)
    M = pred.shape[1]
    return spr, skill, math.sqrt((M+1)/M) * spr / skill

def crps(pred, target, ens_dim=1):
    if len(pred.shape) != len(target.shape):
        target=target.unsqueeze(1)
    num_ens = pred.shape[ens_dim]

    if num_ens == 1:
        crps_estimator = torch.mean(torch.abs(pred - target), dim=ens_dim)

    elif num_ens == 2:
        mean_mae = mae(pred, target).mean(dim=ens_dim)

        # Use simpler estimator
        pair_diffs_term = -0.5 * torch.abs(
            pred.select(ens_dim, 0) - pred.select(ens_dim, 1)
        )

        crps_estimator = mean_mae + pair_diffs_term
    
    elif num_ens < 10:
        # This is the rank-based implementation with O(M*log(M)) compute and
        # O(M) memory. See Zamo and Naveau and WB2 for explanation.
        # For smaller ensemble we can compute all of this directly in memory.
        mean_mae = torch.mean(
            torch.abs(pred - target), dim=ens_dim
        )

        # Ranks start at 1, two argsorts will compute entry ranks
        ranks = pred.argsort(dim=ens_dim).argsort(ens_dim) + 1

        pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
            (num_ens + 1 - 2 * ranks) * pred,
            dim=ens_dim,
        )

        crps_estimator = mean_mae + pair_diffs_term
    else:
        # For large ensembles we batch this over the variable dimension
        crps_res = []
        # Pred is of shape (..., M, D, X, Y)
        for var_i in range(pred.shape[-3]):
            pred_var = pred[..., var_i, :, :]
            target_var = target[..., var_i, :, :]

            mean_mae = torch.mean(
                torch.abs(pred_var - target_var), dim=ens_dim
            )

            # Ranks start at 1, two argsorts will compute entry ranks
            ranks = pred_var.argsort(dim=ens_dim).argsort(ens_dim) + 1

            pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
                (num_ens + 1 - 2 * ranks) * pred_var,
                dim=ens_dim,
            )
            crps_res.append(mean_mae + pair_diffs_term)

        crps_estimator = torch.stack(crps_res, dim=-3)  # (..., D, X, Y)

    return crps_estimator.mean(dim=(-3, -2, -1))

def power_spectrum(x):
    """
    Compute 2D power spectrum for batched input.
    x shape: (..., H, W)
    returns: (B, H, W) averaged over channels
    """
    fft2 = torch.fft.fft2(x, norm="ortho")      # (..., C, H, W)
    fftshift = torch.fft.fftshift(fft2, dim=(-2, -1))
    psd2D = torch.abs(fftshift) ** 2
    return psd2D   # (..., H, W)


def radial_averaged_power(x, ispower=False):
    """
    Radially average 2D power spectrum.
    psd2D shape: (..., H, W) - arbitrary leading dimensions with 2D data at the end
    returns: (..., R) radial profiles, where R = max(H,W)//2
    """
    # Get shape information
    if ispower:
        psd2D = x.cpu()  # Assume x is already the power spectrum
    else:
        psd2D = power_spectrum(x).cpu()  # Work on CPU for bincount
    original_shape = psd2D.shape
    H, W = original_shape[-2], original_shape[-1]

    # Reshape to (-1, H, W) to handle all leading dimensions as one batch dimension
    batch_size = np.prod(original_shape[:-2]) if original_shape[:-2] else 1
    reshaped_psd = psd2D.reshape(batch_size, H, W)

    # Calculate radial coordinates
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).to(psd2D.device)
    r = r.to(torch.int64)

    R = r.max().item() + 1

    # Process each item in the batch
    radial_profiles = []
    for b in range(batch_size):
        tbin = torch.bincount(
            r.flatten(), weights=reshaped_psd[b].flatten(), minlength=R)
        nr = torch.bincount(r.flatten(), minlength=R)
        radial_profiles.append(tbin / torch.clamp(nr, min=1))

    # Stack and reshape back to original leading dimensions
    stacked_profiles = torch.stack(radial_profiles)[:,1:-1]  # (batch_size, R)

    # Reshape back to match original leading dimensions
    if original_shape[:-2]:
        return stacked_profiles.reshape(*original_shape[:-2], stacked_profiles.shape[-1])
    else:
        return stacked_profiles  # Just (R) if input was (H, W)


def radial_integrated_power(x, ispower=False):
    """
    Radially *integrate* (sum) the 2D power spectrum within each radial bin.

    Companion to ``radial_averaged_power``: identical binning, but each bin holds
    the SUM of |F|^2 over the modes in that annulus rather than their per-mode mean.
    This is the spectral *energy* density E(k): because ``power_spectrum`` uses
    ``fft2(norm="ortho")``, Parseval gives  sum_k E(k) ~= sum |x|^2  (real-space
    energy), once the DC bin (k=0) and the outermost incomplete corner bin -- both
    dropped here, matching ``radial_averaged_power`` -- are added back.

    Bin-by-bin it relates to the averaged version exactly as
        radial_integrated_power(x)[..., k] == radial_averaged_power(x)[..., k] * counts[k]
    where ``counts[k]`` is the number of modes at radius k (== RadialPower(H,W).counts).

    psd2D shape: (..., H, W); returns (..., R) radial profiles, R = max(H,W) related
    (DC + corner bin dropped), with arbitrary leading dimensions preserved.
    """
    # Get shape information
    if ispower:
        psd2D = x.cpu()  # Assume x is already the power spectrum
    else:
        psd2D = power_spectrum(x).cpu()  # Work on CPU for bincount
    original_shape = psd2D.shape
    H, W = original_shape[-2], original_shape[-1]

    # Reshape to (-1, H, W) to handle all leading dimensions as one batch dimension
    batch_size = np.prod(original_shape[:-2]) if original_shape[:-2] else 1
    reshaped_psd = psd2D.reshape(batch_size, H, W)

    # Calculate radial coordinates
    cy, cx = H // 2, W // 2
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).to(psd2D.device)
    r = r.to(torch.int64)

    R = r.max().item() + 1

    # Process each item in the batch: SUM power per bin (no division by bin count)
    radial_profiles = []
    for b in range(batch_size):
        tbin = torch.bincount(
            r.flatten(), weights=reshaped_psd[b].flatten(), minlength=R)
        radial_profiles.append(tbin)

    # Stack and drop DC + outermost corner bin (matches radial_averaged_power)
    stacked_profiles = torch.stack(radial_profiles)[:, 1:-1]  # (batch_size, R)

    # Reshape back to match original leading dimensions
    if original_shape[:-2]:
        return stacked_profiles.reshape(*original_shape[:-2], stacked_profiles.shape[-1])
    else:
        return stacked_profiles  # Just (R) if input was (H, W)


class RadialPower(nn.Module):

    def __init__(self, H, W):
        super().__init__()

        cy, cx = H // 2, W // 2

        y, x = torch.meshgrid(
            torch.arange(H),
            torch.arange(W),
            indexing="ij"
        )

        r = torch.sqrt((x - cx) ** 2 + (y - cy) ** 2).long().flatten()
        R = int(r.max()) + 1

        counts = torch.zeros(R)
        counts.scatter_add_(0, r, torch.ones_like(r, dtype=torch.float32))
        counts = counts.clamp(min=1)

        self.R = R

        self.register_buffer("r", r)
        self.register_buffer("counts", counts)

        self.H, self.W = H, W

    def forward(self, x, ispower=False):

        if not ispower:
            psd2D = power_spectrum(x)
        else:
            psd2D = x

        original_shape = psd2D.shape
        H, W = original_shape[-2:]

        lead_shape = original_shape[:-2]
        batch_size = math.prod(lead_shape) if lead_shape else 1

        psd = psd2D.reshape(batch_size, H * W)

        radial_sum = torch.zeros(batch_size, self.R, device=psd.device)

        radial_sum.scatter_add_(1, self.r.expand(batch_size, -1), psd)

        radial_profile = radial_sum / self.counts

        radial_profile = radial_profile[:, 1:-1]

        if lead_shape:
            radial_profile = radial_profile.reshape(*lead_shape, radial_profile.shape[-1])
        else:
            radial_profile = radial_profile.squeeze(0)

        return radial_profile

    def expand(self, radial):
        *lead, R = radial.shape
        B = math.prod(lead) if lead else 1

        radial = radial.reshape(B, R)

        expanded = radial[:, self.r]

        expanded = expanded.reshape(*lead, self.H, self.W)

        return expanded


# ---------------------------------------------------------------------------
# Latitude-weighted skill scores (ERA5).
#
# Grid cells shrink towards the poles, so every ERA5 score in the paper weights
# by cos(latitude), normalised to mean 1. These take model-space tensors with
# latitude on axis -2 and longitude last, and reduce over both spatial axes.
#
#   preds:  (T, M, C, lat, lon)   ensemble forecasts, physical units
#   truth:  (T, C, lat, lon)      ground truth, physical units
#   weights:(lat,)                from compute_area_weights
#
# All three return (T, C). nanmean is used so a diverged member degrades a lead
# rather than poisoning the whole trajectory.
# ---------------------------------------------------------------------------

def _lat_weights(weights: Tensor, ref: Tensor) -> Tensor:
    return weights.to(device=ref.device, dtype=ref.dtype).view(1, 1, -1, 1)


def area_weighted_rmse_all_leads(ens_mean: Tensor, truth: Tensor, weights: Tensor) -> Tensor:
    """RMSE of the ensemble mean -- the "skill" half of the spread-skill ratio."""
    sq_err = (ens_mean - truth) ** 2
    return (sq_err * _lat_weights(weights, sq_err)).nanmean(dim=(-2, -1)).sqrt()


def area_weighted_crps_all_leads(preds: Tensor, truth: Tensor, weights: Tensor) -> Tensor:
    """Fair (M-unbiased) ensemble CRPS: E|X - y| - 0.5 E|X - X'|."""
    M = preds.shape[1]
    term1 = (preds - truth.unsqueeze(1)).abs().mean(dim=1)
    if M > 1:
        # 1/(M(M-1)) rather than 1/M^2: without it CRPS is biased low at small M.
        diff = preds.unsqueeze(1) - preds.unsqueeze(2)
        term2 = diff.abs().sum(dim=(1, 2)) / (M * (M - 1))
    else:
        term2 = torch.zeros_like(term1)
    crps = term1 - 0.5 * term2
    return (crps * _lat_weights(weights, crps)).nanmean(dim=(-2, -1))


def area_weighted_spread_all_leads(preds: Tensor, weights: Tensor) -> Tensor:
    """Ensemble spread: sqrt of the weighted mean population variance.

    Population (1/M) variance; pair with the sqrt((M+1)/M) inflation in
    ``spread_skill_ratio`` to form the calibration diagnostic.
    """
    ens_var = preds.var(dim=1, unbiased=False)
    return (ens_var * _lat_weights(weights, ens_var)).nanmean(dim=(-2, -1)).sqrt()


def spread_skill_ratio(spread: Tensor, rmse: Tensor, n_members: int) -> Tensor:
    """Fair-ensemble spread-skill ratio. 1 is calibrated, below 1 under-dispersed."""
    inflation = math.sqrt((n_members + 1) / n_members)
    return inflation * spread / rmse
