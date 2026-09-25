import torch
from torch import nn, Tensor
from sdecast.autograd import t_dir
from sdecast.nets import SongUNet, attn_config_for_resolution
from sdecast.metrics import RadialPower
import dataclasses
from typing import Any, Optional

@dataclasses.dataclass(frozen=True)
class PosteriorDebugInfo:
    doubt: Any = None
    m_cap: Any = None
    s_cap: Any = None

class Interpolant(nn.Module):
    def __init__(self):
        super().__init__()

        self.cond_dt = False
        self.s_max = 1.0
        self.sample_length = 2
        self.context_size = 2
        self.out = 2

    def get_t(self, t: Tensor, dt: Tensor, ndim: Tensor=4) -> tuple[Tensor, Tensor]:
        t_n = t / dt
        t_n = t_n.view(-1, *([1] * (ndim - 1)))
        t_p = t_n * (1 - t_n)
        return t_n, t_p

    def forward(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        raise NotImplementedError
    
    def linear_interpolant(self, xs: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        mid = self.context_size // 2
        x0, x1 = xs[:, mid - 1], xs[:, mid]
        t_n, _ = self.get_t(t, dt)

        return x0 * (1 - t_n) + x1 * t_n
    
    def _get_time_kwargs(self, expressive: bool, noise_dim: int, channel_mult_emb: int) -> dict:
        if expressive:
            return {
                'embedding_type': 'fourier',
                'channel_mult_noise': 2
            }
        else:
            return {
                'embedding_type': 'linear',
                'emb_activation': nn.Identity(),
                'noise_dim': noise_dim,
                'channel_mult_emb': channel_mult_emb
            }
    

class LinearInterpolant(Interpolant):
    def __init__(self, *args, s_max: float = 1.0, context_size: int = 2, **kwargs):
        super().__init__()
        self.s_max = s_max
        self.context_size = context_size

    def forward(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        m = self.linear_interpolant(xs, t, dt)
        t_n, t_p = self.get_t(t, dt)
        s = t_p * self.s_max

        return m, s, PosteriorDebugInfo(doubt=t_p, m_cap=torch.zeros_like(m), s_cap=s)


class LearnableInterpolant(Interpolant):
    def __init__(self, nx: int, channels: int, hidden_channels: int, s_max: float = 1,
                 cond_dt=False, sample_length=1, m_phi=None, s_phi=None,
                 expressive_time_emb: bool = True, noise_dim: int = 32, channel_mult_emb: int = 1,
                 context_size: int = 2, cond_channels: int = 0, **kwargs):
        super().__init__()
        self.cond_dt = cond_dt
        self.s_max = s_max
        self.sample_length = sample_length
        self.context_size = context_size
        self.cond_channels = cond_channels
        self.out = 2

        time_kwargs = self._get_time_kwargs(expressive_time_emb, noise_dim, channel_mult_emb)
        channel_mult, attn_levels = attn_config_for_resolution(nx)

        self.net = SongUNet(img_resolution=nx, in_channels=self.context_size*channels + cond_channels, out_channels=self.out*channels,
                     encoder_type='residual', decoder_type='standard',
                     resample_filter=[1, 3, 3, 1], model_channels=hidden_channels,
                     circular_padding=(True, True) if cond_channels == 0 else (True, False),
                     channel_mult=channel_mult, attn_levels=attn_levels, time_emb=self.cond_dt,
                     **time_kwargs)


    def forward(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor, PosteriorDebugInfo]:
        res = self.net(xs.flatten(start_dim=1,end_dim=2), t / self.sample_length, class_labels=cond, time_labels=dt / self.sample_length)
        m_res, log_s_res = res.chunk(chunks=2, dim=1)

        t_n, t_p = self.get_t(t, dt)

        s = torch.sigmoid(log_s_res) * self.s_max
        m = self.linear_interpolant(xs, t, dt) + t_p * m_res

        return m, s, PosteriorDebugInfo(doubt=t_p, m_cap=m_res, s_cap=s)


class LearnableInterpolantV2(Interpolant):
    def __init__(self, nx: int, channels: int, hidden_channels: int, s_max: float = 1,
                 cond_dt=False, sample_length=1, m_phi=None, s_phi=None,
                 expressive_time_emb: bool = True, noise_dim: int = 32, channel_mult_emb: int = 1,
                 context_size: int = 2, cond_channels: int = 0, **kwargs):
        super().__init__()
        self.cond_dt = cond_dt
        self.s_max = s_max
        self.sample_length = sample_length
        self.context_size = context_size
        self.cond_channels = cond_channels
        self.out = 2

        time_kwargs = self._get_time_kwargs(expressive_time_emb, noise_dim, channel_mult_emb)
        channel_mult, attn_levels = attn_config_for_resolution(nx)

        self.net = SongUNet(img_resolution=nx, in_channels=self.context_size*channels + cond_channels, out_channels=self.out*channels,
                     encoder_type='residual', decoder_type='standard',
                     resample_filter=[1, 3, 3, 1], model_channels=hidden_channels,
                     circular_padding=(True, True) if cond_channels == 0 else (True, False),
                     channel_mult=channel_mult, attn_levels=attn_levels, time_emb=self.cond_dt,
                     **time_kwargs)


    def forward(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor, PosteriorDebugInfo]:
        res = self.net(xs.flatten(start_dim=1,end_dim=2), t / self.sample_length, class_labels=cond, time_labels=dt / self.sample_length)
        m_res, log_s_res = res.chunk(chunks=2, dim=1)

        t_n, t_p = self.get_t(t, dt)

        s = (torch.exp(log_s_res) ** t_p) * self.s_max
        m = self.linear_interpolant(xs, t, dt) + t_p * m_res

        return m, s, PosteriorDebugInfo(doubt=t_p, m_cap=m_res, s_cap=s)

class FreeInterpolant(Interpolant):
    def __init__(self, nx: int, channels: int, hidden_channels: int, s_max: float = 1, cond_dt=False, sample_length=1, m_phi=None, s_phi=None,
                 noise_dim=32, channel_mult_emb=1, context_size=2, expressive_time_emb: bool = False, cond_channels: int = 0, **kwargs):
        super().__init__()
        self.cond_dt = cond_dt
        self.s_max = s_max
        self.sample_length = sample_length
        self.context_size = context_size
        self.cond_channels = cond_channels
        self.out = 2

        time_kwargs = self._get_time_kwargs(expressive_time_emb, noise_dim, channel_mult_emb)
        channel_mult, attn_levels = attn_config_for_resolution(nx)

        self.net = SongUNet(img_resolution=nx, in_channels=self.context_size*channels + cond_channels, out_channels=self.out*channels,
                     encoder_type='residual', decoder_type='standard', resample_filter=[1, 3, 3, 1],
                     circular_padding=(True, True) if cond_channels == 0 else (True, False),
                     model_channels=hidden_channels, channel_mult=channel_mult, attn_levels=attn_levels, time_emb=cond_dt,
                     **time_kwargs)

    def forward(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor, PosteriorDebugInfo]:
        res = self.net(xs.flatten(start_dim=1,end_dim=2), t / self.sample_length, class_labels=cond, time_labels=dt / self.sample_length)
        m_res, log_s_res = res.chunk(chunks=2, dim=1)

        t_n, t_p = self.get_t(t, dt)

        s = torch.sigmoid(log_s_res) * self.s_max
        m = m_res

        return m, s, PosteriorDebugInfo(doubt=t_p, m_cap=m_res, s_cap=s)

class FixedInterpolant(Interpolant):
    def __init__(self, nx: int, channels: int, hidden_channels: int, s_max: float = 1, cond_dt=False, sample_length=1, m_phi=None, s_phi=None,
                 noise_dim=32, channel_mult_emb=1, context_size=2, expressive_time_emb: bool = False, cond_channels: int = 0, **kwargs):
        super().__init__()
        self.cond_dt = cond_dt
        self.s_max = s_max
        self.sample_length = sample_length
        self.context_size = context_size
        self.cond_channels = cond_channels
        self.out = 2

        time_kwargs = self._get_time_kwargs(expressive_time_emb, noise_dim, channel_mult_emb)
        channel_mult, attn_levels = attn_config_for_resolution(nx)

        self.net = SongUNet(img_resolution=nx, in_channels=self.context_size*channels + cond_channels, out_channels=self.out*channels,
                     encoder_type='residual', decoder_type='standard', resample_filter=[1, 3, 3, 1],
                     circular_padding=(True, True) if cond_channels == 0 else (True, False),
                     model_channels=hidden_channels, channel_mult=channel_mult, attn_levels=attn_levels, time_emb=cond_dt,
                     **time_kwargs)

    def forward(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor, PosteriorDebugInfo]:
        res = self.net(xs.flatten(start_dim=1,end_dim=2), t / self.sample_length, class_labels=cond, time_labels=dt / self.sample_length)
        m_res, log_s_res = res.chunk(chunks=2, dim=1)

        t_n, t_p = self.get_t(t, dt)

        s = torch.sigmoid(log_s_res) * self.s_max * t_p + 0.01
        m = self.linear_interpolant(xs, t, dt) + t_p * m_res

        return m, s, PosteriorDebugInfo(doubt=t_p, m_cap=m_res, s_cap=s)

class NewFixedInterpolant(Interpolant):
    def __init__(self, nx: int, channels: int, hidden_channels: int, s_max: float = 1, s_min: float = 0.01, cond_dt=False, sample_length=1, m_phi=None, s_phi=None,
                 noise_dim=32, channel_mult_emb=1, context_size=2, expressive_time_emb: bool = False, s_channels: int = 1, cond_channels: int = 0, **kwargs):
        super().__init__()
        self.cond_dt = cond_dt

        self.s_max = s_max
        self.s_min = s_min

        self.sample_length = sample_length
        self.context_size = context_size
        self.cond_channels = cond_channels
        self.activation = nn.Sigmoid()

        self.channels = channels
        self.s_channels = s_channels
        self.out = 1 + s_channels

        time_kwargs = self._get_time_kwargs(expressive_time_emb, noise_dim, channel_mult_emb)
        channel_mult, attn_levels = attn_config_for_resolution(nx)

        self.net = SongUNet(img_resolution=nx, in_channels=self.context_size*self.channels + cond_channels, out_channels=self.out*self.channels,
                     encoder_type='residual', decoder_type='standard', resample_filter=[1, 3, 3, 1],
                     circular_padding=(True, True) if cond_channels == 0 else (True, False),
                     model_channels=hidden_channels, channel_mult=channel_mult, attn_levels=attn_levels, time_emb=cond_dt,
                     **time_kwargs)

    def forward(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor, PosteriorDebugInfo]:
        res = self.net(xs.flatten(start_dim=1,end_dim=2), t / self.sample_length, class_labels=cond, time_labels=dt / self.sample_length)

        m_res = res[:, :self.channels]
        s_res = res[:, self.channels:]

        t_n, t_p = self.get_t(t, dt)

        s = self.activation(s_res) * self.s_max * t_p + self.s_min
        m = self.linear_interpolant(xs, t, dt) + t_p * m_res

        return m, s, PosteriorDebugInfo(doubt=t_p, m_cap=m_res, s_cap=s)


class LinearInterpolantWithLearnableS(Interpolant):
    def __init__(self, nx: int, channels: int, hidden_channels: int, s_max: float = 1, s_min: float = 0.01, cond_dt=False, sample_length=1, m_phi=None, s_phi=None,
                 noise_dim=32, channel_mult_emb=1, context_size=2, expressive_time_emb: bool = False, s_channels: int = 1, cond_channels: int = 0, **kwargs):
        super().__init__()
        self.cond_dt = cond_dt

        self.s_max = s_max
        self.s_min = s_min

        self.sample_length = sample_length
        self.context_size = context_size
        self.cond_channels = cond_channels
        self.activation = nn.Sigmoid()

        self.channels = channels
        self.s_channels = s_channels
        self.out = s_channels

        time_kwargs = self._get_time_kwargs(expressive_time_emb, noise_dim, channel_mult_emb)
        channel_mult, attn_levels = attn_config_for_resolution(nx)

        self.net = SongUNet(img_resolution=nx, in_channels=self.context_size*self.channels + cond_channels, out_channels=self.out*self.channels,
                     encoder_type='residual', decoder_type='standard', resample_filter=[1, 3, 3, 1],
                     circular_padding=(True, True) if cond_channels == 0 else (True, False),
                     model_channels=hidden_channels, channel_mult=channel_mult, attn_levels=attn_levels, time_emb=cond_dt,
                     **time_kwargs)

    def forward(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor, PosteriorDebugInfo]:
        s_res = self.net(xs.flatten(start_dim=1, end_dim=2), t / self.sample_length, class_labels=cond, time_labels=dt / self.sample_length)

        t_n, t_p = self.get_t(t, dt)

        s = self.activation(s_res) * self.s_max * t_p + self.s_min
        m = self.linear_interpolant(xs, t, dt)

        return m, s, PosteriorDebugInfo(doubt=t_p, m_cap=torch.zeros_like(m), s_cap=s)


class RadialFixedInterpolant(Interpolant):
    def __init__(self, nx: int, channels: int, hidden_channels: int, s_max: float = 1, s_min: float = 0.0, cond_dt=False, sample_length=1, m_phi=None, s_phi=None,
                 noise_dim=32, channel_mult_emb=1, context_size=2, expressive_time_emb: bool = False, s_channels: int = 1, spectral: bool = False, spatial: bool = False, circular_padding=(True, True), **kwargs):
        super().__init__()
        self.cond_dt = cond_dt

        self.s_max = s_max
        self.s_min = s_min

        self.sample_length = sample_length
        self.context_size = context_size
        self.activation = nn.Sigmoid()

        self.channels = channels
        self.s_channels = s_channels
        self.out = 1 + spatial + s_channels

        time_kwargs = self._get_time_kwargs(expressive_time_emb, noise_dim, channel_mult_emb)
        channel_mult, attn_levels = attn_config_for_resolution(nx)

        self.net = SongUNet(img_resolution=nx, in_channels=self.context_size*self.channels, out_channels=self.out*self.channels,
                     encoder_type='residual', decoder_type='standard', resample_filter=[1, 3, 3, 1],
                     model_channels=hidden_channels, channel_mult=channel_mult, attn_levels=attn_levels, time_emb=cond_dt,
                     circular_padding=circular_padding, **time_kwargs)

        self.spectral = spectral
        self.spatial = spatial

        if spectral:
            self.rad_power = RadialPower(nx, nx)

            self.s_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(s_channels * channels, 256),
                nn.SiLU(),
                nn.Linear(256, channels * self.rad_power.R)
            )

    def forward(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor, PosteriorDebugInfo]:
        res = self.net(xs.flatten(start_dim=1, end_dim=2), t / self.sample_length, time_labels=dt / self.sample_length)

        m_res = res[:, :self.channels]

        t_n, t_p = self.get_t(t, dt)

        if self.spatial:
            s_res = res[:, self.channels:self.channels*2]
            s_spatial = self.activation(s_res) * self.s_max

        if self.spectral:
            s_radial_features = res[:, -self.s_channels*self.channels:]
            s_radial = self.s_head(s_radial_features)
            s_radial = s_radial.view(s_radial.shape[0], self.channels, self.rad_power.R)
            s_radial = torch.exp(s_radial)
            s_radial = self.rad_power.expand(s_radial)

        if self.spatial and self.spectral:
            s = torch.cat([s_spatial * t_p, s_radial], dim=1)
        elif self.spatial:
            s = s_spatial * t_p
        elif self.spectral:
            s = s_radial * t_p
        else:
            raise RuntimeError("At least one of spatial or spectral must be True for RadialFixedInterpolant")

        m = self.linear_interpolant(xs, t, dt) + t_p * m_res

        return m, s, PosteriorDebugInfo(doubt=t_p, m_cap=m_res, s_cap=s)


INTERPOLANT_MAP = {
    "linear": LinearInterpolant,
    "learnable": LearnableInterpolant,
    "learnable_v2": LearnableInterpolantV2,
    "free": FreeInterpolant,
    "fixed": FixedInterpolant,
    "new_fixed": NewFixedInterpolant,
    "linear_learnable_s": LinearInterpolantWithLearnableS,
    "radial_fixed": RadialFixedInterpolant,
}

class SigmaParam(nn.Module):
    def __init__(self, channels: int, *args, **kwargs):
        super().__init__()
        self.s_channels = 1
        self.channels = channels

    def apply_s(self, s: Tensor, eps: Tensor) -> Tensor:
        raise NotImplementedError
    
    def apply_ds(self, s: Tensor, ds: Tensor, eps: Tensor) -> Tensor:
        raise NotImplementedError

    def score(self, s: Any, eps: Tensor, loss_config) -> Tensor:
        raise NotImplementedError

class SpatialDiagSigma(SigmaParam):
    def apply_s(self, s: Tensor, eps: Tensor) -> Tensor:
        z_noise = s * eps
        return z_noise

    def apply_ds(self, s: Tensor, ds: Tensor, eps: Tensor) -> Tensor:
        z_noise = ds * eps
        return z_noise

    def score(self, s: Any, eps: Tensor, loss_config) -> Tensor:
        score = eps / torch.clamp(s, min=loss_config.clamp_s)
        return -score


class SpectralDiagSigma(SigmaParam):
    def apply_s(self, s: Tensor, eps: Tensor) -> Tensor:
        eps = torch.fft.fft2(eps, norm="ortho")
        z_noise = s * eps
        z_noise = torch.fft.ifft2(z_noise, norm="ortho").real        
        return z_noise

    def apply_ds(self, s: Tensor, ds: Tensor, eps: Tensor) -> Tensor:
        eps = torch.fft.fft2(eps, norm="ortho")
        z_noise = ds * eps
        z_noise = torch.fft.ifft2(z_noise, norm="ortho").real
        return z_noise

    def score(self, s: Any, eps: Tensor, loss_config) -> Tensor:
        eps = torch.fft.fft2(eps, norm="ortho")
        score = eps / torch.clamp(s, min=loss_config.clamp_s)
        score = torch.fft.ifft2(score, norm="ortho").real
        return -score

class SpatialSpectralDiagSigma(SigmaParam):
    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(channels, *args, **kwargs)
        self.s_channels = 2
        self.channels = channels

    def apply_s(self, s: Tensor, eps: Tensor) -> Tensor:
        D, C = s[:, :self.channels], s[:, self.channels:]        
        
        res = D * eps
        
        res = torch.fft.fft2(res, norm="ortho")
        res = C * res
        res = torch.fft.ifft2(res, norm="ortho").real
        return res

    def apply_ds(self, s: Tensor, ds: Tensor, eps: Tensor) -> Tensor:
        D, C = s[:, :self.channels], s[:, self.channels:]
        dD, dC = ds[:, :self.channels], ds[:, self.channels:]

        term1 = D * eps
        term1 = torch.fft.fft2(term1, norm="ortho")
        term1 = dC * term1
        term1 = torch.fft.ifft2(term1, norm="ortho").real

        term2 = dD * eps
        term2 = torch.fft.fft2(term2, norm="ortho")
        term2 = C * term2
        term2 = torch.fft.ifft2(term2, norm="ortho").real

        return term1 + term2

    def score(self, s, eps, loss_config):
        D, C = s[:, :self.channels], s[:, self.channels:]   
        D = torch.clamp(D, min=loss_config.clamp_s)
        C = torch.clamp(C, min=loss_config.clamp_s)     

        res = eps
        res = torch.fft.fft2(res, norm="ortho")
        res = res / C
        res = torch.fft.ifft2(res, norm="ortho").real
        res = res / D

        return -res


class SpectralSpatialDiagSigma(SigmaParam):
    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(channels, *args, **kwargs)
        self.s_channels = 2
        self.channels = channels

    def apply_s(self, s: Tensor, eps: Tensor) -> Tensor:
        D, C = s[:, :self.channels], s[:, self.channels:]        

        res = torch.fft.fft2(eps, norm="ortho")
        res = C * res
        res = torch.fft.ifft2(res, norm="ortho").real

        res = D * res

        return res

    def apply_ds(self, s: Tensor, ds: Tensor, eps: Tensor) -> Tensor:
        D, C = s[:, :self.channels], s[:, self.channels:]
        dD, dC = ds[:, :self.channels], ds[:, self.channels:]

        eps_hat = torch.fft.fft2(eps, norm="ortho")

        term1 = C * eps_hat
        term1 = torch.fft.ifft2(term1, norm="ortho").real
        term1 = dD * term1

        term2 = dC * eps_hat
        term2 = torch.fft.ifft2(term2, norm="ortho").real
        term2 = D * term2

        return term1 + term2

    def score(self, s, eps, loss_config):
        D, C = s[:, :self.channels], s[:, self.channels:]   
        D = torch.clamp(D, min=loss_config.clamp_s)
        C = torch.clamp(C, min=loss_config.clamp_s)     

        res = eps
        res = res / D
        res = torch.fft.fft2(res, norm="ortho")
        res = res / C
        res = torch.fft.ifft2(res, norm="ortho").real
        
        return -res
    
class SqrtSpatialSpectralDiagSigma(SigmaParam):
    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(channels, *args, **kwargs)
        self.s_channels = 2
        self.channels = channels

    def apply_s(self, s: Tensor, eps: Tensor) -> Tensor:
        D, C = s[:, :self.channels], s[:, self.channels:]        
        
        res = torch.sqrt(D) * eps
        
        res = torch.fft.fft2(res, norm="ortho")
        res = torch.sqrt(C) * res
        res = torch.fft.ifft2(res, norm="ortho").real
        return res

    def apply_ds(self, s: Tensor, ds: Tensor, eps: Tensor) -> Tensor:
        D, C = s[:, :self.channels], s[:, self.channels:]
        dD, dC = ds[:, :self.channels], ds[:, self.channels:]

        sqrtD = torch.clamp(torch.sqrt(D), min=1e-8)
        sqrtC = torch.clamp(torch.sqrt(C), min=1e-8)

        term1 = sqrtD * eps
        term1 = torch.fft.fft2(term1, norm="ortho")
        term1 = 0.5 * dC / sqrtC * term1
        term1 = torch.fft.ifft2(term1, norm="ortho").real

        term2 = 0.5 * dD / sqrtD * eps
        term2 = torch.fft.fft2(term2, norm="ortho")
        term2 = sqrtC * term2
        term2 = torch.fft.ifft2(term2, norm="ortho").real

        return term1 + term2

    def score(self, s, eps, loss_config):
        D, C = s[:, :self.channels], s[:, self.channels:]   
        D = torch.clamp(D, min=loss_config.clamp_s)
        C = torch.clamp(C, min=loss_config.clamp_s)     

        res = eps
        res = torch.fft.fft2(res, norm="ortho")
        res = res / torch.sqrt(C)
        res = torch.fft.ifft2(res, norm="ortho").real
        res = res / torch.sqrt(D)

        return -res

class SqrtSpectralSpatialDiagSigma(SigmaParam):
    def __init__(self, channels: int, *args, **kwargs):
        super().__init__(channels, *args, **kwargs)
        self.s_channels = 2
        self.channels = channels

    def apply_s(self, s: Tensor, eps: Tensor) -> Tensor:
        D, C = s[:, :self.channels], s[:, self.channels:]        

        res = torch.fft.fft2(eps, norm="ortho")
        res = torch.sqrt(C) * res
        res = torch.fft.ifft2(res, norm="ortho").real

        res = torch.sqrt(D) * res

        return res

    def apply_ds(self, s: Tensor, ds: Tensor, eps: Tensor) -> Tensor:
        D, C = s[:, :self.channels], s[:, self.channels:]
        dD, dC = ds[:, :self.channels], ds[:, self.channels:]

        sqrtD = torch.clamp(torch.sqrt(D), min=1e-8)
        sqrtC = torch.clamp(torch.sqrt(C), min=1e-8)

        eps_hat = torch.fft.fft2(eps, norm="ortho")

        term1 = sqrtC * eps_hat
        term1 = torch.fft.ifft2(term1, norm="ortho").real
        term1 = 0.5 * dD / sqrtD * term1

        term2 = 0.5 * dC / sqrtC * eps_hat
        term2 = torch.fft.ifft2(term2, norm="ortho").real
        term2 = sqrtD * term2

        return term1 + term2

    def score(self, s, eps, loss_config):
        D, C = s[:, :self.channels], s[:, self.channels:]   
        D = torch.clamp(D, min=loss_config.clamp_s)
        C = torch.clamp(C, min=loss_config.clamp_s)     

        res = eps
        res = res / torch.sqrt(D)
        res = torch.fft.fft2(res, norm="ortho")
        res = res / torch.sqrt(C)
        res = torch.fft.ifft2(res, norm="ortho").real
        
        return -res


SIGMA_MAP = {
    "spatial_diag": SpatialDiagSigma,
    "spectral_diag": SpectralDiagSigma,
    "spectral_spatial_diag": SpectralSpatialDiagSigma,
    "spatial_spectral_diag": SpatialSpectralDiagSigma,
    "sqrt_spectral_spatial_diag": SqrtSpectralSpatialDiagSigma,
    "sqrt_spatial_spectral_diag": SqrtSpatialSpectralDiagSigma,
}

class PosteriorAffine(nn.Module):
    def __init__(self,  nx, channels, hidden_channels, *args, interpolant_type="learnable", sigma_type="diag", **kwargs):
        super().__init__()

        self.debug_info = PosteriorDebugInfo()
        
        self.sigma = SIGMA_MAP[sigma_type](channels)
        self.interpolant = INTERPOLANT_MAP[interpolant_type](nx=nx,
            channels=channels,
            hidden_channels=hidden_channels,
            s_channels=self.sigma.s_channels,
            **kwargs
        )
    def get_coeffs(self, xs: Tensor, t: Tensor, dt: Tensor, cond: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        m, s, debug_info = self.interpolant(xs, t, dt, cond)
        self.debug_info = debug_info

        return m, s

    def forward(
            self,
            ctx: Tensor,
            t: Tensor,
            dt: Tensor,
            cond: Optional[Tensor] = None,
            return_t_dir: bool = False
    ) -> tuple[Tensor, Tensor] | tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        if return_t_dir:
            def f(t_in: Tensor) -> Tensor:
               return self.get_coeffs(ctx, t_in, dt, cond)
            return t_dir(f, t)
        else:
            return self.get_coeffs(ctx, t, dt, cond)

    def sample(self, m, s, return_eps=False, eps=None) -> Tensor:
        if eps is None:
            eps = torch.randn_like(m)

        z_noise = self.sigma.apply_s(s, eps)
        
        if return_eps:
            return m + z_noise, eps
        return m + z_noise

    def q_dz(self, dm, ds, s, eps):
        dz_noise = self.sigma.apply_ds(s, ds, eps)
        return dm + dz_noise
    
    def q_score(self, s, eps, loss_config):
        score = self.sigma.score(s, eps, loss_config)
        return score
