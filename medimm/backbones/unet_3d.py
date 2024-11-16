from typing import Optional, List, Union, Tuple, Sequence, Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from medimm.layers.norm import LayerNorm3d


class UNet3dConfig(NamedTuple):
    in_channels: int = 1
    channels: Sequence[int] = (8, 32, 128, 512)
    depths: Sequence[Union[int, Tuple[int, int]]] = ((1, 1), (2, 2), (4, 4), 8)
    stem_stride: Union[int, Tuple[int, int, int]] = 1
    stem_kernel_size: Optional[Union[int, Tuple[int, int, int]]] = 7
    stem_padding: Union[int, Tuple[int, int, int]] = 3
    drop_path_rate: float = 0.0


class UNet3dOutput(NamedTuple):
    feature_pyramid: List[torch.Tensor]


class Stem3d(nn.Module):
    def __init__(self, config: UNet3dConfig) -> None:
        super().__init__()

        stride = _to_tuple(config.stem_stride)
        if config.stem_kernel_size is not None:
            kernel_size = _to_tuple(config.stem_kernel_size)
        else:
            kernel_size = config.stem_stride
        padding = _to_tuple(config.stem_padding)

        self.conv = nn.Conv3d(config.in_channels, config.channels[0] - 1, kernel_size, stride, padding)
        self.norm = LayerNorm3d(config.channels[0] - 1)
        self.stride = stride

        self.mask_token = nn.Parameter(torch.zeros(config.channels[0] - 1))
        nn.init.trunc_normal_(self.mask_token)

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            x = self.conv(image)
            x = self.norm(x)
            n, _, h, w, d = x.shape
            mask = torch.ones((n, 1, h, w, d), dtype=x.dtype, device=x.device)
            x = torch.cat([x, mask], dim=1)
            return x
        else:
            mask = mask.unsqueeze(1)
            x = self.conv(image * mask)
            mask = F.max_pool3d(mask, kernel_size=self.stride)
            x = x * mask + self.mask_token.view(-1, 1, 1, 1) * (1 - mask)
            x = self.norm(x)
            x = torch.cat([x, mask], dim=1)
            return x


class UNetBlock3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels)
        self.norm = LayerNorm3d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv3d(in_channels, out_channels, kernel_size=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ = x
        x = self.conv_1(x)
        x = self.norm(x)
        self.act(x)
        x = self.conv_2(x)
        x = x + self.shortcut(input_)
        return x


class UNetStage3d(nn.Module):
    def __init__(
            self,
            channels: int,
            depth: int,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            UNetBlock3d(channels, channels)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        return x


class UNet3d(nn.Module):
    def __init__(
            self,
            config: UNet3dConfig
    ) -> None:
        super().__init__()

        self.stem = Stem3d(config)
        self.left_stages = nn.ModuleList([])
        self.down_convs = nn.ModuleList([])
        self.up_convs = nn.ModuleList([])
        self.skip_connections = nn.ModuleList([])
        self.right_stages = nn.ModuleList([])
        for c_1, c_2, (d_1, d_2) in zip(config.channels, config.channels[1:], config.depths):
            self.left_stages.append(UNetStage3d(c_1, d_1))
            self.down_convs.append(nn.Conv3d(c_1, c_2, kernel_size=2, stride=2))
            self.up_convs.append(
                nn.Sequential(
                    nn.Conv3d(c_2, c_1, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='trilinear')
                )
            )
            self.skip_connections.append(nn.Conv3d(c_1, c_1, kernel_size=1))
            self.right_stages.append(UNetStage3d(c_1, d_2))
        self.bottom_stage = UNetStage3d(config.channels[-1], config.depths[-1])

        self.config = config
        self.min_input_size = tuple(s * 2 ** len(self.down_convs) for s in _to_tuple(config.stem_stride))

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        if any(image.shape[i] < self.min_input_size[i] for i in [-3, -2, -1]):
            raise ValueError(f"Input's spatial size {x.shape[-3:]} is less than {self.max_stride}.")

        if mask is not None and mask.dtype != image.dtype:
            raise TypeError("``mask`` must have the same dtype as input image ``x``")

        # stem
        x = self.stem(image, mask)

        # UNet's down path
        feature_pyramid = []
        for i in range(len(self.down_convs)):
            x = self.left_stages[i](x)
            feature_pyramid.append(x)
            x = self.down_convs[i](x)

        # UNet's bottom layers
        x = self.bottom_stage(x)
        feature_pyramid.append(x)

        # UNet's up path
        for i in reversed(range(len(self.up_convs))):
            x = self.up_convs[i](x)
            y = self.skip_connections[i](feature_pyramid[i])
            x = _crop_and_pad_to(x, y)
            x = x + y
            x = self.right_stages[i](x)
            feature_pyramid[i] = x

        return UNet3dOutput(feature_pyramid)


def _crop_and_pad_to(x: torch.Tensor, other: torch.Tensor, pad_mode: str = 'replicate') -> torch.Tensor:
    assert x.ndim == other.ndim == 5

    # crop
    x = x[(..., *map(slice, other.shape[-3:]))]

    # pad
    pad = []
    for dim in [-1, -2, -3]:
        pad += [0, max(other.shape[dim] - x.shape[dim], 0)]
    x = F.pad(x, pad, mode=pad_mode)

    return x


def _to_tuple(int_or_seq: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(int_or_seq, int):
        return (int_or_seq, int_or_seq, int_or_seq)
    else:
        assert len(int_or_seq) == 3

        return tuple(int_or_seq)
