# Code adapted from:
# Elucidating the Design Space of Diffusion-Based Generative Models (EDM)
# Tero Karras, Miika Aittala, Timo Aila, Samuli Laine
# https://github.com/NVlabs/edm

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch.nn.functional import silu

# ----------------------------------------------------------------------------
# Unified routine for initializing weights and biases.


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform':
        return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':
        return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform':
        return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':
        return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

# ----------------------------------------------------------------------------
# Fully-connected layer.


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(
            mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init(
            [out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init(
            [out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x

# ----------------------------------------------------------------------------
# Convolutional layer with optional up/downsampling.


class Conv2d(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, kernel, bias=True, up=False, down=False,
                 resample_filter=[1, 1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
                 circular_padding=(True, False),
                 ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels *
                           kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init(
            [out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init(
            [out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

        self.circ_pad_x = circular_padding[0]
        self.circ_pad_y = circular_padding[1]

    def circ_pad(self, x, pad):
        pad_x, pad_y = pad * self.circ_pad_x, pad * self.circ_pad_y
        pad_func = torch.nn.CircularPad2d((pad_x, pad_x, pad_y, pad_y))

        return pad_func(x) if pad else x

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(
            x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            r_pad = max(f_pad - w_pad, 0)
            x = self.circ_pad(x, r_pad)
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile(
                [self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=(r_pad * (1 + 2 * self.circ_pad_y), r_pad * (1 + 2 * self.circ_pad_x)))

            r_pad = max(w_pad - f_pad, 0)
            x = self.circ_pad(x, r_pad)
            x = torch.nn.functional.conv2d(x, w, padding=(r_pad * (not self.circ_pad_y), r_pad * (not self.circ_pad_x)))
        elif self.fused_resample and self.down and w is not None:
            r_pad = w_pad+f_pad
            x = self.circ_pad(x, r_pad)
            x = torch.nn.functional.conv2d(x, w, padding=(r_pad * (not self.circ_pad_y), r_pad * (not self.circ_pad_x)))
            x = torch.nn.functional.conv2d(
                x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = self.circ_pad(x, f_pad)
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile(
                    [self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=(f_pad * (1 + 2 * self.circ_pad_y), f_pad * (1 + 2 * self.circ_pad_x)))
            if self.down:
                x = self.circ_pad(x, f_pad)
                x = torch.nn.functional.conv2d(x, f.tile(
                    [self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=(f_pad * (not self.circ_pad_y), f_pad * (not self.circ_pad_x)))
            if w is not None:
                x = self.circ_pad(x, w_pad)
                x = torch.nn.functional.conv2d(x, w, padding=(w_pad * (not self.circ_pad_y), w_pad * (not self.circ_pad_x)))
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x

# ----------------------------------------------------------------------------
# Group normalization.


class GroupNorm(torch.nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels //
                              min_channels_per_group)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        x = torch.nn.functional.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(
            x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
        return x

# ----------------------------------------------------------------------------
# Attention weight computation, i.e., softmax(Q^T * K).
# Performs all computation using FP32, but uses the original datatype for
# inputs/outputs/gradients to conserve memory.


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k /
                         np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(
            torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32),
                          db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32),
                          db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk

# ----------------------------------------------------------------------------
# Unified U-Net block with optional up/downsampling and self-attention.
# Represents the union of all features employed by the DDPM++, NCSN++, and
# ADM architectures.


class UNetBlock(torch.nn.Module):
    def __init__(self,
                 in_channels, out_channels, emb_channels, up=False, down=False, attention=False,
                 num_heads=None, channels_per_head=64, dropout=0, skip_scale=1, eps=1e-5,
                 resample_filter=[1, 1], resample_proj=False, adaptive_scale=True,
                 init=dict(), init_zero=dict(init_weight=0), init_attn=None,
                 circular_padding=(True, False),
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = 0 if not attention else num_heads if num_heads is not None else out_channels // channels_per_head
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel=3, up=up, down=down, resample_filter=resample_filter, **init, circular_padding=circular_padding)
        self.affine = Linear(in_features=emb_channels,
                             out_features=out_channels*(2 if adaptive_scale else 1), **init)
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(in_channels=out_channels,
                            out_channels=out_channels, kernel=3, **init_zero, circular_padding=circular_padding)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel=kernel, up=up, down=down, resample_filter=resample_filter, **init, circular_padding=circular_padding)

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(in_channels=out_channels, out_channels=out_channels*3,
                              kernel=1, **(init_attn if init_attn is not None else init), circular_padding=circular_padding)
            self.proj = Conv2d(in_channels=out_channels,
                               out_channels=out_channels, kernel=1, **init_zero, circular_padding=circular_padding)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.adaptive_scale:
            scale, shift = params.chunk(chunks=2, dim=1)
            x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
        else:
            x = silu(self.norm1(x.add_(params)))

        x = self.conv1(torch.nn.functional.dropout(
            x, p=self.dropout, training=self.training))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        if self.num_heads:
            q, k, v = self.qkv(self.norm2(x)).reshape(
                x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
            w = AttentionOp.apply(q, k)
            a = torch.einsum('nqk,nck->ncq', w, v)
            x = self.proj(a.reshape(*x.shape)).add_(x)
            x = x * self.skip_scale
        return x

# ----------------------------------------------------------------------------
# Timestep embedding used in the DDPM++ and ADM architectures.


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels //
                             2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

# ----------------------------------------------------------------------------
# Timestep embedding used in the NCSN++ architecture.


class FourierEmbedding(torch.nn.Module):
    def __init__(self, num_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(num_channels // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x
    
class LinearNoiseEmbedding(torch.nn.Module):
    def __init__(self, noise_dim=32):
        super(LinearNoiseEmbedding, self).__init__()
        self.linear = Linear(1, noise_dim)

    def forward(self, noise):
        noise_level_encoding = self.linear(noise.unsqueeze(1))  # NOTE this is for it to work in the same way as Fourier embedding.
        if noise_level_encoding.dim() == 1:
            noise_level_encoding = noise_level_encoding.unsqueeze(
                0)  # Add back batch dimension if missing
        return noise_level_encoding

# ----------------------------------------------------------------------------
# Channel multipliers and attention levels for a SongUNet at a given spatial
# resolution. Attention is gated per *level* (level index into channel_mult),
# so the levels below reproduce the configurations that have been validated:
#   * SQG and ERA5 32x64 (nx == 64): the original 3-level net with attention at
#     level 1 (the nx//2 resolution the old resolution-based gate selected).
#   * ERA5 121x240 (nx == 240): a deeper 4-level net with attention at the two
#     deepest levels, matching the proven DhariwalUNet config (attn_levels=(2,3)).


def attn_config_for_resolution(nx):
    if nx == 240:
        return [1, 2, 3, 4], (2, 3)
    return [2, 2, 2], (1,)

# ----------------------------------------------------------------------------
# Reimplementation of the DDPM++ and NCSN++ architectures from the paper
# "Score-Based Generative Modeling through Stochastic Differential
# Equations". Equivalent to the original implementation by Song et al.,
# available at https://github.com/yang-song/score_sde_pytorch


class SongUNet(torch.nn.Module):
    def __init__(self,
                 # Image resolution at input/output.
                 img_resolution,
                 # Number of color channels at input.
                 in_channels,
                 # Number of color channels at output.
                 out_channels,
                 # Number of class labels, 0 = unconditional.
                 label_dim=0,
                 # Augmentation label dimensionality, 0 = no augmentation.
                 augment_dim=0,

                 # Base multiplier for the number of channels.
                 model_channels=128,
                 # Per-resolution multipliers for the number of channels.
                 channel_mult=[1, 2, 2, 2],
                 # Multiplier for the dimensionality of the embedding vector.
                 channel_mult_emb=4,
                 # Number of residual blocks per resolution.
                 num_blocks=4,
                 # Levels (indices into channel_mult) with self-attention.
                 attn_levels=(),
                 # Dropout probability of intermediate activations.
                 dropout=0.10,
                 # Dropout probability of class labels for classifier-free guidance.
                 label_dropout=0,

                 # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                 embedding_type='fourier',
                 # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                 channel_mult_noise=1,
                 # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                 encoder_type='standard',
                 # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                 decoder_type='standard',
                 # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
                 resample_filter=[1, 1],

                 time_emb=False,            # If time labels, 0 = no time labels.
                 noise_dim=32,
                 emb_activation=silu,       # Activation for embedding layers.
                 circular_padding=(True, False),
                 ):
        assert embedding_type in ['fourier', 'positional', 'linear']
        assert encoder_type in ['standard', 'skip', 'residual']
        assert decoder_type in ['standard', 'skip']

        super().__init__()
        self.label_dropout = label_dropout
        emb_channels = model_channels * channel_mult_emb
        noise_channels = model_channels * channel_mult_noise
        init = dict(init_mode='xavier_uniform')
        init_zero = dict(init_mode='xavier_uniform', init_weight=1e-5)
        init_attn = dict(init_mode='xavier_uniform', init_weight=np.sqrt(0.2))

        # Mapping.
        if embedding_type == 'positional':
            self.map_noise = PositionalEmbedding(
                num_channels=noise_channels, endpoint=True)
        elif embedding_type == 'fourier':
            self.map_noise = FourierEmbedding(num_channels=noise_channels)
        else:
            self.map_noise = LinearNoiseEmbedding(noise_dim=noise_dim)
            noise_channels = noise_dim
            emb_channels = noise_dim

        self.map_time = (PositionalEmbedding(num_channels=noise_channels, endpoint=True) if embedding_type ==
                         'positional' else FourierEmbedding(num_channels=noise_channels)) if time_emb else None
        self.map_label = Linear(
            in_features=label_dim, out_features=noise_channels, **init) if label_dim else None
        self.map_augment = Linear(
            in_features=augment_dim, out_features=noise_channels, bias=False, **init) if augment_dim else None
        self.map_layer0 = Linear(
            in_features=noise_channels, out_features=emb_channels, **init)
        self.map_layer1 = Linear(
            in_features=emb_channels, out_features=emb_channels, **init)
        self.emb_activation = emb_activation
        self.circular_padding = circular_padding

        block_kwargs = dict(
            emb_channels=emb_channels, num_heads=1, dropout=dropout, skip_scale=np.sqrt(0.5), eps=1e-6,
            resample_filter=resample_filter, resample_proj=True, adaptive_scale=False,
            init=init, init_zero=init_zero, init_attn=init_attn, circular_padding=circular_padding
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        caux = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            if level == 0:
                cin = cout
                cout = model_channels
                self.enc[f'{res}x{res}_conv'] = Conv2d(
                    in_channels=cin, out_channels=cout, kernel=3, **init, circular_padding=circular_padding)
            else:
                self.enc[f'{res}x{res}_down'] = UNetBlock(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs)
                if encoder_type == 'skip':
                    self.enc[f'{res}x{res}_aux_down'] = Conv2d(
                        in_channels=caux, out_channels=caux, kernel=0, down=True, resample_filter=resample_filter, circular_padding=circular_padding)
                    self.enc[f'{res}x{res}_aux_skip'] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=1, **init, circular_padding=circular_padding)
                if encoder_type == 'residual':
                    self.enc[f'{res}x{res}_aux_residual'] = Conv2d(
                        in_channels=caux, out_channels=cout, kernel=3, down=True, resample_filter=resample_filter, fused_resample=True, **init, circular_padding=circular_padding)
                    caux = cout
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                attn = (level in attn_levels)
                self.enc[f'{res}x{res}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
        skips = [block.out_channels for name,
                 block in self.enc.items() if 'aux' not in name]

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            if level == len(channel_mult) - 1:
                self.dec[f'{res}x{res}_in0'] = UNetBlock(
                    in_channels=cout, out_channels=cout, attention=True, **block_kwargs)
                self.dec[f'{res}x{res}_in1'] = UNetBlock(
                    in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f'{res}x{res}_up'] = UNetBlock(
                    in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                attn = (idx == num_blocks and level in attn_levels)
                self.dec[f'{res}x{res}_block{idx}'] = UNetBlock(
                    in_channels=cin, out_channels=cout, attention=attn, **block_kwargs)
            if decoder_type == 'skip' or level == 0:
                if decoder_type == 'skip' and level < len(channel_mult) - 1:
                    self.dec[f'{res}x{res}_aux_up'] = Conv2d(
                        in_channels=out_channels, out_channels=out_channels, kernel=0, up=True, resample_filter=resample_filter, circular_padding=circular_padding)
                self.dec[f'{res}x{res}_aux_norm'] = GroupNorm(
                    num_channels=cout, eps=1e-6)
                self.dec[f'{res}x{res}_aux_conv'] = Conv2d(
                    in_channels=cout, out_channels=out_channels, kernel=3, **init_zero, circular_padding=circular_padding)

    def forward(self, x, noise_labels, class_labels=None, time_labels=None, augment_labels=None):
        # Mapping.
        emb = self.map_noise(noise_labels)
        emb = emb.reshape(
            emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)  # swap sin/cos
        if self.map_label is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * \
                    (torch.rand([x.shape[0], 1], device=x.device)
                     >= self.label_dropout).to(tmp.dtype)
            emb = emb + \
                self.map_label(tmp * np.sqrt(self.map_label.in_features))
        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)
        if self.map_time is not None and time_labels is not None:
            tmp = self.map_time(time_labels)
            tmp = tmp.reshape(
                tmp.shape[0], 2, -1).flip(1).reshape(*tmp.shape)  # swap sin/cos
            if self.training and self.label_dropout:
                tmp = tmp * \
                    (torch.rand([x.shape[0], 1], device=x.device)
                     >= self.label_dropout).to(tmp.dtype)
            emb = emb + tmp

        emb = self.emb_activation(self.map_layer0(emb))
        emb = self.emb_activation(self.map_layer1(emb))

        # Conditioning.
        if class_labels is not None:
            tmp = class_labels
            if self.training and self.label_dropout:
                tmp = tmp * \
                    (torch.rand([x.shape[0], 1, 1, 1], device=x.device)
                     >= self.label_dropout).to(tmp.dtype)
            x = torch.cat((x, tmp), dim=1)

        # Encoder.
        skips = []
        aux = x
        for name, block in self.enc.items():
            if 'aux_down' in name:
                aux = block(aux)
            elif 'aux_skip' in name:
                x = skips[-1] = x + block(aux)
            elif 'aux_residual' in name:
                x = skips[-1] = aux = (x + block(aux)) / np.sqrt(2)
            else:
                x = block(x, emb) if isinstance(block, UNetBlock) else block(x)
                skips.append(x)

        # Decoder.
        aux = None
        tmp = None
        for name, block in self.dec.items():
            if 'aux_up' in name:
                aux = block(aux)
            elif 'aux_norm' in name:
                tmp = block(x)
            elif 'aux_conv' in name:
                tmp = block(silu(tmp))
                aux = tmp if aux is None else tmp + aux
            else:
                if x.shape[1] != block.in_channels:
                    skip = skips.pop()
                    if skip.shape[-2:] != x.shape[-2:]:
                        x = torch.nn.functional.interpolate(
                            x, size=skip.shape[-2:], mode='bilinear', align_corners=False
                        )
                    x = torch.cat([x, skip], dim=1)
                x = block(x, emb)
        return aux
