from typing import NamedTuple, Optional, List, Union, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3dConfig(NamedTuple):
    in_channels: int = 1
    channels: int = (32, 64, 128, 256, 512, 1024)
    depths: Sequence[int] = (1, 1, 2, 2, 4, 4)


class UNet3dOutput(NamedTuple):
    feature_maps: torch.Tensor
    feature_pyramid: List[torch.Tensor]


class UNetBlock3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: Union[int, Tuple[int, int, int]] = 1
    ) -> None:
        super().__init__()

        self.conv_1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm_1 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act_1 = nn.LeakyReLU(inplace=True)
        self.conv_2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm_2 = nn.InstanceNorm3d(out_channels, affine=True)
        self.act_2 = nn.LeakyReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ = x
        x = self.conv_1(x)
        x = self.norm_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x += self.shortcut(input_)
        x = self.act_2(x)
        return x


class UNet3d(nn.Module):
    def __init__(self, config: UNet3dConfig) -> None:
        super().__init__()

        self.encoder_stages = nn.ModuleList([])
        self.decoder_ups = nn.ModuleList([])
        self.decoder_stages = nn.ModuleList([])

        self.encoder_stages.append(
            nn.Sequential(
                UNetBlock3d(config.in_channels + 1, config.channels[0], stride=1),
                *[UNetBlock3d(config.channels[0], config.channels[0]) for _ in range(config.depths[0] - 1)]
            )
        )
        for i in range(len(config.channels) - 1):
            self.encoder_stages.append(
                nn.Sequential(
                    UNetBlock3d(config.channels[i], config.channels[i + 1], stride=2),
                    *[UNetBlock3d(config.channels[i + 1], config.channels[i + 1]) for _ in range(config.depths[i + 1] - 1)]
                )
            )
            self.decoder_ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv3d(config.channels[i + 1], config.channels[i], kernel_size=1),
                )
            )
            self.decoder_stages.append(
                nn.Sequential(
                    UNetBlock3d(config.channels[i] * 2, config.channels[i]),
                    *[UNetBlock3d(config.channels[i], config.channels[i]) for _ in range(config.depths[i] - 1)]
                )
            )

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> UNet3dOutput:
        if any(image.shape[i] < 2 ** (len(self.encoder_stages) - 1) for i in [-3, -2, -1]):
            raise ValueError(f"Input's spatial size {x.shape[-3:]} is less than {self.max_stride}.")

        if mask is None:
            n, _, h, w, d = image.shape
            mask = torch.ones((n, h, w, d), dtype=image.dtype, device=image.device)
        elif mask.dtype != image.dtype:
            raise TypeError("``mask`` must have the same dtype as input image ``x``")
        mask = mask.unsqueeze(1)
        x = torch.cat([image * mask, mask], dim=1)

        # encoder
        feature_pyramid = []
        for stage in self.encoder_stages:
            x = stage(x)
            feature_pyramid.append(x)

        # decoder
        for i in reversed(range(len(self.decoder_stages))):
            x = self.decoder_ups[i](x)
            y = feature_pyramid[i]
            x = crop_and_pad_to(x, y)
            x = torch.cat([x, y], dim=1)
            x = self.decoder_stages[i](x)
            feature_pyramid[i] = x

        return UNet3dOutput(x, feature_pyramid)


def crop_and_pad_to(x: torch.Tensor, other: torch.Tensor, pad_mode: str = 'replicate') -> torch.Tensor:
    assert x.ndim == other.ndim == 5

    if x.shape == other.shape:
        return x

    # crop
    x = x[(..., *map(slice, other.shape[-3:]))]

    # pad
    pad = []
    for dim in [-1, -2, -3]:
        pad += [0, max(other.shape[dim] - x.shape[dim], 0)]
    x = F.pad(x, pad, mode=pad_mode)

    return x


# class UNet3dWithConvNextBackbone:
#     """After UNet3d encoder, enrich feature pyramid with ConvNext features.
#     """
#     pass


# class UNet3dWithViTBackbone:
#     """After UNet3d encoder, enrich feature pyramid with ViT features.
#     """
#     pass
