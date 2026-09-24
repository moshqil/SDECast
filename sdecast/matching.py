"""The SDE-Cast model container.

Holds the prior SDE (``p_sde`` -- the learnt drift and volatility that inference
integrates) and the learnable posterior interpolant (``q_affine``). Training-time
loss terms are not part of this release; what remains is the module structure the
checkpoints were saved against, plus ``posterior_drift`` for inspecting the
interpolant's implied drift.
"""
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor

from sdecast.autograd import grad
from sdecast.metrics import RadialPower
from sdecast.sde import SDE


@dataclass(frozen=True)
class LossConfig:
    """Numerical clamps shared by the posterior score. Defaults match training."""
    clamp_s: float = 1e-3
    clamp_g2: float = 1e-7


class MatchingSDE_SQG(nn.Module):
    def __init__(
            self,
            p_sde: SDE,
            q_affine: nn.Module,
            p_observe: nn.Module,
            sample_length: int,
            nx: int,
            clamp=None,
            regularization_factor=None,
            spectra_path: Optional[str] = None,
            drift_spectra_path: Optional[str] = None,
            divergence_target: Optional[float] = None,
            h_hours: float = 3.0,
    ):
        super().__init__()
        self.p_sde = p_sde
        self.q_affine = q_affine
        self.p_observe = p_observe
        self.clamp = clamp
        self.sample_length = sample_length

        def g2_dg2(g2_in, z):
            g2 = g2_in(z)
            return g2, torch.zeros_like(g2)

        state_independent_vol = (
            self.p_sde.vol_type in ("const", "channel_const", "full_const", "spectral_radial")
            or isinstance(self.p_sde.vol_type, float)
        )
        self.g2_dg2 = g2_dg2 if state_independent_vol else grad

        self.rad_power = RadialPower(nx, nx)

        # Spectral-loss references are training-only buffers. They are registered so
        # checkpoints trained with them load cleanly.
        if spectra_path is not None:
            self.register_buffer("log_r_ref", torch.load(spectra_path).unsqueeze(0))
        else:
            self.log_r_ref = None
        if drift_spectra_path is not None:
            self.register_buffer("drift_log_r_ref", torch.load(drift_spectra_path).unsqueeze(0))
        else:
            self.drift_log_r_ref = None

        self.register_buffer("area_weights", None, persistent=False)

    def set_area_weights(self, area_weights: Tensor):
        self.register_buffer("area_weights", area_weights.float(), persistent=False)

    def posterior_drift(self, xs: Tensor, t: Tensor, dt: Tensor,
                        cond: Optional[Tensor] = None,
                        loss_config: Optional[LossConfig] = None):
        """Drift the posterior bridge implies at interpolation time ``t``.

        ``xs`` is (B, context, C, H, W) -- the bridge endpoints. Returns
        ``(q_drift, z)``: the posterior drift and the sampled interpolant state.
        At the optimum this matches ``p_sde.drift(z, t, cond)``, which is what
        makes the learnt prior drift meaningful.
        """
        loss_config = loss_config or LossConfig()
        (m, s), (dm, ds) = self.q_affine(xs, t, dt, cond, return_t_dir=True)
        z, eps = self.q_affine.sample(m, s, return_eps=True)
        q_dz = self.q_affine.q_dz(dm, ds, s, eps)
        q_score = self.q_affine.q_score(s, eps, loss_config)

        g2, d_g2 = self.g2_dg2(lambda z_in: self.p_sde.vol(z_in, t, cond) ** 2, z)
        q_drift = q_dz + 0.5 * self.p_sde.apply(g2, q_score) + 0.5 * d_g2
        return q_drift.detach(), z.detach()
