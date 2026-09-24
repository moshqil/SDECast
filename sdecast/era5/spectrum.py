"""Kinetic-energy power spectrum for ERA5 10 m wind via spherical harmonics.

Ported from PhysMetrics.Weather (`_ke_spectrum_spharm`), which uses pyshtools'
Driscoll-Healy spherical-harmonic expansion. The KE power at spherical-harmonic
degree ``l`` is

    E(l) = 0.5 * sum_{m=0..l} (|u_lm|^2 + |v_lm|^2)

with ``u_lm, v_lm`` the real spherical-harmonic coefficients of the (physical,
m/s) zonal and meridional wind.

Two entry points:
  * ``ke_spectrum_uv(u, v)``        -- pure kernel on 2-D physical-unit fields.
  * ``ke_spectrum_from_state(...)`` -- ERA5 adapter: selects u10/v10 from a
    model-space *normalized* state, denormalizes, orients N->S, and delegates.

Note: this is a *spectral* diagnostic. Do NOT cosine-area-weight the inputs --
the spherical-harmonic transform already integrates over the sphere. (Cosine
weights belong only to the grid-space RMSE path in sdecast.metrics.)
"""

from __future__ import annotations

import numpy as np

# Channel layout of the model state (must match ERA5Dataset.variables /
# evaluate.VAR_NAMES). Hard-coded here to keep this module import-light.
VAR_NAMES = ['t2m', 'u10', 'v10', 'z500', 't850']
U10_IDX = 1
V10_IDX = 2
assert VAR_NAMES[U10_IDX] == 'u10' and VAR_NAMES[V10_IDX] == 'v10'


def ke_spectrum_uv(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """KE spectrum E(l) of a 2-D wind field via pyshtools SHExpandDH.

    Faithful port of PhysMetrics.Weather `_ke_spectrum_spharm`.

    Parameters
    ----------
    u, v : ndarray, shape (nlat, nlon)
        Physical-unit wind components, N->S latitude order.

    Returns
    -------
    wavenumber, energy : ndarray
        Spherical-harmonic degree ``l`` (0..lmax) and KE at each degree.
    """
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

    # pyshtools needs even-sized grids
    if nlat % 2 != 0:
        u, v = u[:-1, :], v[:-1, :]
        nlat -= 1
    if nlon % 2 != 0:
        u, v = u[:, :-1], v[:, :-1]
        nlon -= 1

    # SHExpandDH requires nlon == 2*nlat (sampling=2) or nlon == nlat (sampling=1).
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
    # Vectorized equivalent of the reference's `for l: for m:` power sum. The DH
    # coefficient array is (2, lmax+1, lmax+1): [0]=cosine, [1]=sine. Sine
    # coefficients at m=0 are exactly 0, so summing them in is a no-op; the
    # upper-triangular (m>l) entries are 0 too. Sum over the order axis m.
    pw = u_c[0] ** 2 + v_c[0] ** 2 + u_c[1] ** 2 + v_c[1] ** 2  # (lmax+1, lmax+1)
    energy = 0.5 * pw.sum(axis=1)                                # (lmax+1,)

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
    """KE spectrum of (u10, v10) from a model-space normalized ERA5 state.

    Parameters
    ----------
    state : ndarray or torch.Tensor, shape (C, lat, lon)
        Single model-space *normalized* state. Latitude on axis -2, longitude on
        axis -1.
    mean, std : torch.Tensor (1, C, 1, 1) or array-like
        Per-channel normalization stats (as returned by load_stats). Used to
        denormalize u/v back to physical units (m/s).
    u_idx, v_idx : int
        Channel indices of u10 / v10 (defaults 1 / 2).
    lat : 1-D array-like or None
        The latitude values (degrees) for the state's lat axis. pyshtools'
        Driscoll-Healy grid expects North->South (row 0 = north pole). The grid
        is asymmetric about the equator, so E(l) is NOT flip-invariant -- the
        orientation genuinely matters. If ``lat`` is given, the field is oriented
        to N->S automatically (flip iff ``lat`` is ascending / South->North).
        Recommended: always pass ``dataset.lat``.
    flip_lat : bool
        Fallback used only when ``lat`` is None: force-flip the latitude axis.

    Returns
    -------
    wavenumber, energy : ndarray
    """
    state = _to_numpy(state)
    if state.ndim != 3:
        raise ValueError(f"Expected (C, lat, lon) state, got shape {state.shape}")

    mean = np.asarray(_to_numpy(mean), dtype=np.float64).reshape(-1)
    std = np.asarray(_to_numpy(std), dtype=np.float64).reshape(-1)

    # Denormalize to physical m/s (model space is ~unit variance; u and v have
    # different stds, so skipping this would distort their relative KE).
    u = state[u_idx].astype(np.float64) * std[u_idx] + mean[u_idx]
    v = state[v_idx].astype(np.float64) * std[v_idx] + mean[v_idx]

    # Orient to N->S for pyshtools (row 0 = north pole).
    if lat is not None:
        lat_arr = np.asarray(_to_numpy(lat), dtype=np.float64).reshape(-1)
        to_ns = bool(lat_arr[0] < lat_arr[-1])   # ascending S->N -> flip
    else:
        to_ns = bool(flip_lat)
    if to_ns:
        u = u[::-1, :]
        v = v[::-1, :]

    return ke_spectrum_uv(u, v)
