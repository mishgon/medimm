from typing import NamedTuple, List, Sequence, Optional, Literal, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import hf_hub_download

from medimm.layers.norm import LayerNorm2d
from medimm.layers.layer_scale import LayerScale2d


class UNet2dConfig(NamedTuple):
    in_channels: int = 1
    hidden_channels: int = (16, 32, 64, 128, 256, 512)
    depths: Sequence[int] = (1, 1, 2, 2, 4, 4)
    num_attn_heads: int = 8
    time_embed_dim: int = 1024


class UNet2dOutput(NamedTuple):
    feature_pyramid: List[torch.Tensor]


class TimeEmbed(nn.Module):
    def __init__(self, time_embed_dim: int = 1024):
        super().__init__()

        self.time_embed_dim = time_embed_dim

        self.proj = nn.Sequential(
            nn.Linear(256, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.ndim == 1

        t = t.float() * 1000.0
        num_freqs = 128
        t = t[:, None] * torch.exp(-math.log(10_000) * torch.arange(num_freqs).to(t) / num_freqs)
        t = torch.cat([t.cos(), t.sin()], dim=1)

        t = self.proj(t)

        return t


class ResBlock2d(nn.Module):
    def __init__(self, channels: int, time_embed_dim: int) -> None:
        super().__init__()

        self.norm_1 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels)
        self.act_1 = nn.SiLU(inplace=True)
        self.conv_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_embed_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels)
        )
        self.norm_2 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels)
        self.act_2 = nn.SiLU(inplace=True)
        self.conv_2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ls = LayerScale2d(channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.norm_1(x)
        h = self.act_1(h)
        h = self.conv_1(h)
        h = h + self.time_embed_proj(t)[:, :, None, None]
        h = self.norm_2(h)
        h = self.act_2(h)
        h = self.conv_2(h)
        h = self.ls(h)
        return x + h


class AttentionBlock2d(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()

        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.ls = LayerScale2d(channels)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        N = H * W
        h = self.norm(x)
        h = h.flatten(2).movedim(-1, 1)
        q, k, v = self.qkv(h).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, N, C)
        h = self.proj(h)
        h = h.movedim(1, -1).unflatten(-1, (H, W))
        h = self.ls(h)
        return x + h


class Stage(nn.Module):
    def __init__(
            self,
            channels: int,
            depth: int,
            time_embed_dim: int,
            attn: bool,
            num_attn_heads: Optional[int] = None
    ) -> None:
        super().__init__()

        self.depth = depth
        self.attn = attn

        self.res_blocks = nn.ModuleList([ResBlock2d(channels, time_embed_dim) for _ in range(depth)])
        if attn:
            self.attn_blocks = nn.ModuleList([AttentionBlock2d(channels, num_attn_heads) for _ in range(depth - 1)])

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        for i in range(self.depth):
            x = self.res_blocks[i](x, t)
            if self.attn and i != self.depth - 1:
                self.attn_blocks[i](x)
        return x


class UNet2d(nn.Module):
    def __init__(self, config: UNet2dConfig) -> None:
        super().__init__()

        self.config = config

        self.stem_conv = nn.Conv2d(config.in_channels, config.hidden_channels[0], kernel_size=3, padding=1)

        self.time_embed = TimeEmbed(config.time_embed_dim)

        self.encoder_stages = nn.ModuleList([])
        self.down_convs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.fuse_convs = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])

        self.encoder_stages.append(
            Stage(
                channels=config.hidden_channels[0],
                depth=config.depths[0],
                time_embed_dim=config.time_embed_dim,
                attn=False
            )
        )
        for i in range(len(config.hidden_channels) - 1):
            self.down_convs.append(
                nn.Conv2d(
                    in_channels=config.hidden_channels[i],
                    out_channels=config.hidden_channels[i + 1],
                    kernel_size=2,
                    stride=2
                )
            )
            self.encoder_stages.append(
                Stage(
                    channels=config.hidden_channels[i + 1],
                    depth=config.depths[i + 1],
                    time_embed_dim=config.time_embed_dim,
                    attn=(i + 1 >= 4),
                    num_attn_heads=config.num_attn_heads
                )
            )
            self.ups.append(
                nn.Upsample(scale_factor=2, mode='nearest'),
            )
            self.fuse_convs.append(
                nn.Conv2d(
                    in_channels=config.hidden_channels[i + 1] + config.hidden_channels[i],
                    out_channels=config.hidden_channels[i],
                    kernel_size=1
                )
            )
            self.decoder_stages.append(
                Stage(
                    channels=config.hidden_channels[i],
                    depth=config.depths[i],
                    time_embed_dim=config.time_embed_dim,
                    attn=(i >= 4),
                    num_attn_heads=config.num_attn_heads
                )
            )

    def forward(self, image: torch.Tensor, t: Union[float, torch.Tensor] = 1.0) -> UNet2dOutput:
        if any(image.shape[i] < 2 ** (len(self.encoder_stages) - 1) for i in [-2, -1]):
            raise ValueError(f"Input's spatial size {image.shape[-3:]} is less than {self.max_stride}.")

        # first conv
        x = self.stem_conv(image)

        # time embed
        if isinstance(t, float):
            t = torch.full(size=(len(x),), fill_value=t, dtype=x.dtype, device=x.device)
        t = self.time_embed(t)

        # encoder
        feature_pyramid = []
        for i in range(len(self.encoder_stages)):
            x = self.encoder_stages[i](x, t)
            feature_pyramid.append(x)
            if i != len(self.encoder_stages) - 1:
                x = self.down_convs[i](x)

        # decoder
        for i in reversed(range(len(self.decoder_stages))):
            x = self.ups[i](x)
            y = feature_pyramid[i]
            x = crop_and_pad_to(x, y)
            x = torch.cat([x, y], dim=1)
            x = self.fuse_convs[i](x)
            x = self.decoder_stages[i](x, t)
            feature_pyramid[i] = x

        return UNet2dOutput(feature_pyramid)


def crop_and_pad_to(x: torch.Tensor, other: torch.Tensor, pad_mode: str = 'replicate') -> torch.Tensor:
    assert x.ndim == other.ndim == 4

    if x.shape == other.shape:
        return x

    # crop
    x = x[(..., *map(slice, other.shape[-2:]))]

    # pad
    pad = []
    for dim in [-1, -2]:
        pad += [0, max(other.shape[dim] - x.shape[dim], 0)]
    x = F.pad(x, pad, mode=pad_mode)

    return x


UNet2dSize = Literal['tiny', 'small', 'base', 'large', 'xlarge', 'xxlarge']


def unet2d(size: UNet2dSize, **kwargs) -> UNet2d:
    return globals()[f'unet2d_{size}'](**kwargs)


def unet2d_tiny(**kwargs) -> UNet2d:
    config = UNet2dConfig(hidden_channels=(8, 16, 32, 64, 128, 256), num_attn_heads=4, time_embed_dim=512)
    if kwargs:
        config = config._replace(**kwargs)
    model = UNet2d(config)
    return model


def unet2d_small(**kwargs) -> UNet2d:
    config = UNet2dConfig(hidden_channels=(12, 24, 48, 96, 192, 384), num_attn_heads=6, time_embed_dim=768)
    if kwargs:
        config = config._replace(**kwargs)
    model = UNet2d(config)
    return model


def unet2d_base(**kwargs) -> UNet2d:
    config = UNet2dConfig(hidden_channels=(16, 32, 64, 128, 256, 512), num_attn_heads=8, time_embed_dim=1024)
    if kwargs:
        config = config._replace(**kwargs)
    model = UNet2d(config)
    return model


def unet2d_large(**kwargs) -> UNet2d:
    config = UNet2dConfig(hidden_channels=(24, 48, 96, 192, 384, 768), num_attn_heads=12, time_embed_dim=1536)
    if kwargs:
        config = config._replace(**kwargs)
    model = UNet2d(config)
    return model


def unet2d_xlarge(**kwargs) -> UNet2d:
    config = UNet2dConfig(hidden_channels=(32, 64, 128, 256, 512, 1024), num_attn_heads=16, time_embed_dim=2048)
    if kwargs:
        config = config._replace(**kwargs)
    model = UNet2d(config)
    return model


def unet2d_xxlarge(**kwargs) -> UNet2d:
    config = UNet2dConfig(hidden_channels=(48, 96, 192, 384, 768, 1536), num_attn_heads=24, time_embed_dim=3072)
    if kwargs:
        config = config._replace(**kwargs)
    model = UNet2d(config)
    return model
