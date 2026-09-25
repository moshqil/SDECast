import torch
from torch import nn, Tensor
from torch import distributions as D
from sdecast.sde import SDE
from sdecast.nets import SongUNet, attn_config_for_resolution
from sdecast.metrics import RadialPower


class SpectralVol(nn.Module):
    def __init__(self, channels, rad_power):
        super().__init__()
        self.channels = channels
        self.rad_power = rad_power
        self.R = rad_power.R

        self.log_power = nn.Parameter(
            torch.zeros(channels, self.R)
        )

    def forward(self):
        power = torch.exp(self.log_power)
        sigma = torch.sqrt(power)
        sigma = self.rad_power.expand(sigma)
        return sigma

class PriorSDE(SDE):
    def __init__(self, nx: int, channels: int, hidden_channels: int, vol_type="local", max_sigma=0.1, cond_channels: int = 0,
                 use_true_drift: bool = False, h_hours: float = 3.0):
        super().__init__()

        self.cond_channels = cond_channels
        self.use_true_drift = use_true_drift
        self.h_hours = float(h_hours)
        ny = nx // 2 + 1 if nx == 240 else nx // 2

        channel_mult, attn_levels = attn_config_for_resolution(nx)

        if use_true_drift:
            if cond_channels != 0:
                raise ValueError("use_true_drift is SQG-only; cond_channels must be 0 (no conditioning).")
            from sdecast.sqg.torch_solver import SQGPrior
            from sdecast.sqg.constants import data_std
            self.drift_physical = SQGPrior(std=data_std[0])
        else:
            self.drift_net = SongUNet(img_resolution=nx, in_channels=channels + cond_channels, out_channels=channels,
                         embedding_type='fourier', encoder_type='residual', decoder_type='standard',
                         channel_mult_noise=2, resample_filter=[1, 3, 3, 1], model_channels=hidden_channels,
                         circular_padding=(True, True) if cond_channels == 0 else (True, False),
                         channel_mult=channel_mult, attn_levels=attn_levels
                         )


        self.vol_type = vol_type
        self.max_sigma=max_sigma
        if vol_type == "const":
            self.log_sigma = nn.Parameter(torch.zeros(1))
        elif vol_type == "channel_const":
            self.log_sigma = nn.Parameter(torch.zeros(channels))
        elif vol_type == "full_const":
            self.log_sigma = nn.Parameter(torch.zeros(channels, ny, nx))
        elif isinstance(vol_type, float):
            self.register_buffer('log_sigma', torch.tensor(vol_type))

        elif vol_type == "local":
            self.vol_net = nn.Sequential(
                nn.Linear(1, channels),
                nn.Softplus(),
                nn.Linear(channels, 1),
            )

        elif vol_type == "channel_learnt":
            self._vol_in_dim = 1 + cond_channels
            vol_hidden = 32
            self.vol_net = nn.Sequential(
                nn.Conv2d(channels * self._vol_in_dim, channels * vol_hidden,
                          kernel_size=1, groups=channels),
                nn.SiLU(),
                nn.Conv2d(channels * vol_hidden, channels,
                          kernel_size=1, groups=channels),
            )

        elif vol_type == "unet":
            self.vol_net = SongUNet(img_resolution=nx, in_channels=channels + cond_channels, out_channels=channels,
                     embedding_type='fourier', encoder_type='residual', decoder_type='standard',
                     channel_mult_noise=2, resample_filter=[1, 3, 3, 1], model_channels=hidden_channels,
                     circular_padding=(True, True) if cond_channels == 0 else (True, False),
                     channel_mult=channel_mult, attn_levels=attn_levels
                     )
        elif vol_type == "spectral_radial":
            self.rad_power = RadialPower(nx, nx)
            self.vol_net = SpectralVol(channels, self.rad_power)
        else:
            raise ValueError(f"Unknown vol_type: {vol_type}")
        

    def drift(self, z: Tensor, t: Tensor, cond=None, *args) -> Tensor:
        if self.use_true_drift:
            return self.drift_physical(z, t) * (self.h_hours / 3.0)
        t = torch.zeros_like(t)
        return self.drift_net(z, t, class_labels=cond)

    def vol(self, z: Tensor, t: Tensor, cond=None, *args) -> Tensor:
        if self.vol_type == "const" or isinstance(self.vol_type, float):
            sigma = torch.sigmoid(self.log_sigma) if self.vol_type == "const" else self.log_sigma
            g = sigma * torch.ones_like(z)

        elif self.vol_type == "channel_const":
            sigma = torch.sigmoid(self.log_sigma)
            g = sigma[None, :, None, None] * torch.ones_like(z)

        elif self.vol_type == "full_const":
            sigma = torch.sigmoid(self.log_sigma)
            g = sigma[None, :, :, :] * torch.ones_like(z)

        elif self.vol_type == "local":
            z_flat = z.unsqueeze(-1)
            g = self.vol_net(z_flat).squeeze(-1)

        elif self.vol_type == "channel_learnt":
            B, C, H, W = z.shape
            z_per_ch = z.unsqueeze(2)
            if self.cond_channels > 0:
                cond_per_ch = cond.unsqueeze(1).expand(B, C, self.cond_channels, H, W)
                x_in = torch.cat([z_per_ch, cond_per_ch], dim=2)
            else:
                x_in = z_per_ch
            x_in = x_in.reshape(B, C * self._vol_in_dim, H, W)
            g = self.vol_net(x_in)

        elif self.vol_type == "unet":
            t = torch.zeros_like(t)
            g = self.vol_net(z, t, class_labels=cond)
        elif self.vol_type == "spectral_radial":
            g = self.vol_net()
        else:
            raise RuntimeError("Invalid vol_type")

        g = g * self.max_sigma
        return g
    
    def apply(self, A: Tensor, b: Tensor) -> Tensor:
        if self.vol_type == "spectral_radial":
            b = torch.fft.fft2(b, norm="ortho")
            Ab = A * b
            res = torch.fft.ifft2(Ab, norm="ortho").real        
        else:
            res = A * b
        return res
    
    def inverse(self, A: Tensor, b: Tensor) -> Tensor:
        if self.vol_type == "spectral_radial":
            b = torch.fft.fft2(b, norm="ortho")
            Ab = b / A
            res = torch.fft.ifft2(Ab, norm="ortho").real        
        else:
            res = b / A
        return res


class PriorObservation(nn.Module):
    def __init__(self, noise_std: float):
        super().__init__()

        self.noise_std = noise_std

    def get_coeffs(self, z: Tensor) -> tuple[Tensor, Tensor]:
        m = z
        s = torch.ones_like(m) * self.noise_std
        return m, s

    def forward(self, z: Tensor) -> D.Distribution:
        m, s = self.get_coeffs(z)
        return D.Independent(D.Normal(m, s), 3)
