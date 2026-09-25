from __future__ import annotations

import numpy as np

VAR_NAMES = ['t2m', 'u10', 'v10', 'z500', 't850']
U10_IDX = 1
V10_IDX = 2
assert VAR_NAMES[U10_IDX] == 'u10' and VAR_NAMES[V10_IDX] == 'v10'


def ke_spectrum_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyshtools as pysh
    except ImportError:
        raise ImportError("pyshtools is required for the ERA5 KE spectrum. "
                          "Install it with:  pip install -e '.[spectrum]'")

    u = np.asarray(u, dtype=np.float64).squeeze()
    v = np.asarray(v, dtype=np.float64).squeeze()

    if u.ndim != 2:
        raise ValueError(f"Expected 2-D wind fields, got u.shape = {u.shape}")

    nlat, nlon = u.shape

    if nlat % 2 != 0:
        u, v = u[:-1, :], v[:-1, :]
        nlat -= 1
    if nlon % 2 != 0:
        u, v = u[:, :-1], v[:, :-1]
        nlon -= 1

    if nlon == 2 * nlat:
        sampling = 2
        lmax = nlat // 2 - 1
    elif nlon == nlat:
        sampling = 1
        lmax = nlat // 2 - 1
    else:
        raise ValueError(
            f"Grid ({nlat}, {nlon}) is not a valid Driscoll-Healy grid. "
            f"Spectral analysis requires nlon == nlat or nlon == 2*nlat."
        )

    u_c = pysh.expand.SHExpandDH(u, sampling=sampling, lmax_calc=lmax)
    v_c = pysh.expand.SHExpandDH(v, sampling=sampling, lmax_calc=lmax)

    wavenumber = np.arange(lmax + 1)
    pw = u_c[0] ** 2 + v_c[0] ** 2 + u_c[1] ** 2 + v_c[1] ** 2
    energy = 0.5 * pw.sum(axis=1)

    return wavenumber, energy


def _to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def ke_spectrum_from_state(
    state,
    mean,
    std,
    u_idx: int = U10_IDX,
    v_idx: int = V10_IDX,
    lat=None,
    flip_lat: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    state = _to_numpy(state)
    if state.ndim != 3:
        raise ValueError(f"Expected (C, lat, lon) state, got shape {state.shape}")

    mean = np.asarray(_to_numpy(mean), dtype=np.float64).reshape(-1)
    std = np.asarray(_to_numpy(std), dtype=np.float64).reshape(-1)

    u = state[u_idx].astype(np.float64) * std[u_idx] + mean[u_idx]
    v = state[v_idx].astype(np.float64) * std[v_idx] + mean[v_idx]

    if lat is not None:
        lat_arr = np.asarray(_to_numpy(lat), dtype=np.float64).reshape(-1)
        to_ns = bool(lat_arr[0] < lat_arr[-1])
    else:
        to_ns = bool(flip_lat)
    if to_ns:
        u = u[::-1, :]
        v = v[::-1, :]

    return ke_spectrum_uv(u, v)
