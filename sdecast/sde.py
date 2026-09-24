"""SDE interface and Euler-Maruyama solvers.

``solve_sde`` is the unconditional sampler (SQG); ``solve_sde_time_dependent_cond``
assembles the per-step ERA5 conditioning and supports ``keep_steps`` so long
roll-outs need not buffer every frame. Pass ``stochastic=False`` to integrate the
probability-flow ODE (drift only) instead.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence

import torch
from torch import nn, Tensor
from tqdm import tqdm


class SDE(nn.Module, ABC):
    @abstractmethod
    def drift(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def vol(self, z: Tensor, t: Tensor, *args: Any) -> Tensor:
        raise NotImplementedError

    def forward(self, z: Tensor, t: Tensor, *args: Any) -> tuple[Tensor, Tensor]:
        drift = self.drift(z, t, *args)
        vol = self.vol(z, t, *args)
        return drift, vol

@torch.no_grad()
def solve_sde(
        sde: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int,
        stochastic=True,
        cond: Tensor = None
) -> Tensor:
    B = z.shape[0]
    tt = torch.linspace(ts, tf, n_steps + 1, device=z.device)[:-1]

    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5

    path = torch.empty((n_steps + 1, *z.shape), device="cpu")
    path[0] = z.cpu()

    for i in tqdm(range(n_steps)):
        t = tt[i].expand(B)
        f, g = sde(z, t, cond)
        w = torch.randn_like(z, device=z.device)
        z = z + f * dt

        if stochastic:
            z = z + dt_2 * sde.apply(g, w)

        path[i + 1] = z.cpu()

    return path


@torch.no_grad()
def solve_sde_time_dependent_cond(
        sde: Callable[[Tensor, Tensor, Tensor], tuple[Tensor, Tensor]],
        z: Tensor,
        ts: float,
        tf: float,
        n_steps: int,
        stochastic: bool = True,
        static_cond: Optional[Tensor] = None,
        time_cond: Optional[Sequence[Tensor]] = None,
        keep_steps: Optional[Sequence[int]] = None,
) -> Tensor:
    """SDE solver for ERA5 where the conditioning at step i is
    concat([time_cond[i], static_cond]) along the channel dim.

    If ``keep_steps`` is given (step indices in ``[0, n_steps]``), only those
    frames are stored, returned in the requested order. This avoids holding the
    full ``(n_steps + 1, *z.shape)`` trajectory on the host, which is
    prohibitive at high resolution. Defaults to keeping every step.
    """
    B = z.shape[0]
    tt = torch.linspace(ts, tf, n_steps + 1, device=z.device)[:-1]

    dt = (tf - ts) / n_steps
    dt_2 = abs(dt) ** 0.5

    if keep_steps is None:
        keep_steps = range(n_steps + 1)
    # step index -> output positions (a step may appear more than once)
    positions: dict[int, list[int]] = {}
    keep_list = [int(s) for s in keep_steps]
    for p, s in enumerate(keep_list):
        positions.setdefault(s, []).append(p)

    path = torch.empty((len(keep_list), *z.shape), device="cpu")

    def store(step: int, value: Tensor) -> None:
        if step in positions:
            value_cpu = value.cpu()
            for p in positions[step]:
                path[p] = value_cpu

    store(0, z)

    # Upstream called load_spatial_embeddings() with no argument here, which is a
    # TypeError; conditioning is required for this solver, so refuse explicitly.
    if static_cond is None:
        raise ValueError(
            "solve_sde_time_dependent_cond requires static_cond. Build it with "
            "sdecast.era5.cond.build_static_era5_cond(spatial_embeddings_grid(lat, lon), masks)."
        )
    static_cond = static_cond.to(z.device)

    for i in tqdm(range(n_steps)):
        t = tt[i].expand(B)

        if time_cond is not None:
            from sdecast.era5.cond import assemble_era5_cond
            step_time_cond = time_cond[i].to(z.device)
            cond = assemble_era5_cond(step_time_cond, static_cond)
        else:
            cond = static_cond

        f, g = sde(z, t, cond)
        w = torch.randn_like(z, device=z.device)
        z = z + f * dt

        if stochastic:
            z = z + dt_2 * sde.apply(g, w)

        store(i + 1, z)

    return path
