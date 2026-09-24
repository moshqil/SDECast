"""Ground-truth SQG drift.

The trained prior SDE learns an (autonomous) drift ``f(z)`` on the *standardized*
state ``z = PV / data_std`` in *train-time* units (1 unit = ``h`` hours, the
training lead). The physical "ground truth" for that drift is the deterministic
SQG tendency ``dPV/dt`` produced by the spectral solver in ``sdecast.sqg.solver``
(``SQG.gettend``), converted into the same standardized / train-time units so the
two are directly comparable:

    f_gt(z) = irfft2( gettend( rfft2(z * data_std) ) ) * (h * 3600) / data_std

``gettend`` is the continuous RHS (nonlinear advection + thermal relaxation, plus
Ekman if enabled); it excludes the per-step hyperdiffusion filter, which is a
numerical sub-grid dissipation applied as an integrating factor inside
``timestep``, not part of the physical drift.

The solver runs on CPU/numpy (pyfftw falls back to ``numpy.fft``); only ``numpy``
and ``scipy`` are needed beyond the inference venv.
"""
from __future__ import annotations

import numpy as np
import torch

from sdecast.sqg.solver import SQG, rfft2, irfft2
from sdecast.sqg.constants import data_std as _DATA_STD  # [2660, 2660]

# SQG solver parameters for N=64, taken verbatim from scripts/generate_sqg_data.py
# (the branch `args.N == 64`) and gen_data_mac.sh (`--N 64`). These determine the
# tendency (U -> equilibrium jet, tdiab -> relaxation, L/H/f/nsq -> inversion).
SQG_PARAMS_N64 = dict(
    nsq=1.0e-4,
    f=1.0e-4,
    U=30.0,
    H=10.0e3,
    r=0.0,                 # dek=0 -> no Ekman damping
    tdiab=10.0 * 86400,    # 10-day thermal relaxation
    L=20.0 * (np.sqrt(1.0e-4) * 10.0e3 / 1.0e-4),  # 20 * Rossby radius = 2e7 m
    dt=1200.0,             # 20-min step (only affects hyperdiff, not gettend)
    diff_order=8,
    diff_efold=86400.0,
    dealias=True,
    symmetric=True,
    threads=1,
    precision="single",
    tstart=0,
)


def build_sqg_model(nx: int = 64, params: dict | None = None) -> SQG:
    """Instantiate the SQG spectral solver matching the data-generation config.

    The initial ``pv`` only sizes the grid; ``gettend`` is evaluated at explicit
    states and never mutates the model, so the model is reusable across calls.
    """
    p = {**SQG_PARAMS_N64, **(params or {})}
    pv0 = np.zeros((2, nx, nx), dtype=np.float64)  # only used to set N
    return SQG(pv0, **p)


def ground_truth_drift(
    z_std,
    model: SQG | None = None,
    h_hours: float = 3.0,
    data_std=None,
):
    """Deterministic SQG drift in standardized / train-time units.

    Args:
        z_std: ``(..., 2, nx, nx)`` standardized state(s), torch tensor or ndarray.
        model: a reusable ``SQG`` solver (built with ``build_sqg_model`` if None).
        h_hours: hours per train-time unit (the training lead ``h``; default 3).
        data_std: per-channel standardization std (defaults to sdecast.sqg.constants).

    Returns:
        torch.Tensor of the same shape as ``z_std`` holding ``dz_std/dt_train``.
    """
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
        pv = (flat[i] * ds).astype(np.float32)                 # standardized -> model PV units
        dpvspec_dt = model.gettend(rfft2(pv, threads=model.threads))
        dpv_dt = irfft2(dpvspec_dt, threads=model.threads)     # dPV/dt [PV units / second]
        out[i] = dpv_dt * sec_per_unit / ds                    # -> dz_std / dt_train
    out = out.reshape(arr.shape)

    res = torch.from_numpy(out.astype(np.float32))
    return res.to(device) if is_torch else res
