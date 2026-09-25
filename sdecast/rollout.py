from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from sdecast.sde import solve_sde, solve_sde_time_dependent_cond


def rollout_sqg(prior_sde, x0: Tensor, n_ens: int = 1, ts: float = 0.0,
                tf: float = 1.0, steps_per_unit_time: int = 100,
                stochastic: bool = True) -> Tensor:
    if n_ens > 1:
        x0 = x0.expand(n_ens, -1, -1, -1).contiguous()
    n_steps = max(1, int(round(steps_per_unit_time * (tf - ts))))
    return solve_sde(prior_sde, x0, ts, tf, n_steps, stochastic=stochastic)


def rollout_era5(prior_sde, x0: Tensor, static_cond: Tensor, n_ens: int = 1,
                 ts: float = 0.0, tf: float = 1.0, steps_per_unit_time: int = 64,
                 init_time_days: float | Tensor = 0.0,
                 keep_steps: Optional[Sequence[int]] = None,
                 stochastic: bool = True) -> Tensor:
    from sdecast.era5.cond import temporal_embeddings

    if n_ens > 1:
        x0 = x0.expand(n_ens, -1, -1, -1).contiguous()
    B = x0.shape[0]
    n_steps = max(1, int(round(steps_per_unit_time * (tf - ts))))

    sde_time = torch.linspace(ts, tf, n_steps + 1, device=x0.device)[:-1]
    init_t = torch.as_tensor(init_time_days, dtype=torch.float32, device=x0.device)

    if init_t.dim() == 0:
        t_days = init_t + (sde_time - ts) / 24.0
        time_cond = temporal_embeddings(t_days).view(n_steps, 1, 4, 1, 1).expand(n_steps, B, 4, 1, 1)
    else:
        t_days = init_t.unsqueeze(0) + (sde_time - ts).unsqueeze(1) / 24.0
        time_emb = temporal_embeddings(t_days.reshape(-1)).view(n_steps, B, 4)
        time_cond = time_emb.view(n_steps, B, 4, 1, 1).expand(n_steps, B, 4, 1, 1)

    return solve_sde_time_dependent_cond(
        prior_sde, x0, ts, tf, n_steps, stochastic=stochastic,
        static_cond=static_cond, time_cond=time_cond, keep_steps=keep_steps,
    )
