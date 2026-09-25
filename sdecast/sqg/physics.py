from __future__ import annotations

import numpy as np
import torch

from sdecast.sqg.solver import SQG, rfft2, irfft2
from sdecast.sqg.constants import data_std as _DATA_STD

SQG_PARAMS_N64 = dict(
    nsq=1.0e-4,
    f=1.0e-4,
    U=30.0,
    H=10.0e3,
    r=0.0,
    tdiab=10.0 * 86400,
    L=20.0 * (np.sqrt(1.0e-4) * 10.0e3 / 1.0e-4),
    dt=1200.0,
    diff_order=8,
    diff_efold=86400.0,
    dealias=True,
    symmetric=True,
    threads=1,
    precision="single",
    tstart=0,
)


def build_sqg_model(nx: int = 64, params: dict | None = None) -> SQG:
    p = {**SQG_PARAMS_N64, **(params or {})}
    pv0 = np.zeros((2, nx, nx), dtype=np.float64)
    return SQG(pv0, **p)


def ground_truth_drift(
    z_std,
    model: SQG | None = None,
    h_hours: float = 3.0,
    data_std=None,
):
    if model is None:
        model = build_sqg_model(nx=z_std.shape[-1])
    ds = np.asarray(_DATA_STD if data_std is None else data_std, dtype=np.float64).reshape(2, 1, 1)
    sec_per_unit = float(h_hours) * 3600.0

    is_torch = torch.is_tensor(z_std)
    device = z_std.device if is_torch else None
    arr = (z_std.detach().cpu().numpy() if is_torch else np.asarray(z_std)).astype(np.float64)

    nx = arr.shape[-1]
    flat = arr.reshape(-1, 2, arr.shape[-2], nx)
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        pv = (flat[i] * ds).astype(np.float32)
        dpvspec_dt = model.gettend(rfft2(pv, threads=model.threads))
        dpv_dt = irfft2(dpvspec_dt, threads=model.threads)
        out[i] = dpv_dt * sec_per_unit / ds
    out = out.reshape(arr.shape)

    res = torch.from_numpy(out.astype(np.float32))
    return res.to(device) if is_torch else res
