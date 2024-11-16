from typing import Optional, List, Union, Tuple, Sequence, Any, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath

from medimm.layers.norm import LayerNorm3d, GlobalResponseNorm3d
from medimm.layers.layer_scale import LayerScale3d


class ConvNeXt3dConfig(NamedTuple):
    in_channels: int = 1
    channels: Sequence[int] = (96, 192, 384, 768)
    depths: Sequence[Union[int, Tuple[int, int]]] = (3, 3, 9, 3)
    stem_stride: Union[int, Tuple[int, int, int]] = 4
    stem_kernel_size: Optional[Union[int, Tuple[int, int, int]]] = None
    stem_padding: Union[int, Tuple[int, int, int]] = 0
    drop_path_rate: float = 0.0


class ConvNeXt3dOutput(NamedTuple):
    feature_pyramid: List[torch.Tensor]
    pooled_features: torch.Tensor


class Stem3d(nn.Module):
    def __init__(self, config: ConvNeXt3dConfig) -> None:
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


class ConvNeXtBlock3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_factor: float = 4.0,
            kernel_size: int = 3,
            dropout_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            grn: bool = False,
            layer_scale: bool = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(in_channels * hidden_factor)
        self.conv_1 = nn.Conv3d(in_channels, in_channels, kernel_size, padding='same', groups=in_channels)
        self.norm = LayerNorm3d(in_channels)
        self.conv_2 = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm3d(hidden_channels) if grn else nn.Identity()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv_3 = nn.Conv3d(hidden_channels, out_channels, kernel_size=1)
        self.layerscale = LayerScale3d(out_channels, init_values=1e-6) if layer_scale else nn.Identity()
        self.drop_path = DropPath(drop_path_rate)

        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_ = x
        x = self.conv_1(x)
        x = self.norm(x)
        x = self.conv_2(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.dropout(x)
        x = self.conv_3(x)
        x = self.layerscale(x)
        x = self.drop_path(x)
        x = x + self.shortcut(input_)
        return x


class ConvNeXtStage3d(nn.Module):
    def __init__(
            self,
            channels: int,
            depth: int,
            drop_path_rates: Optional[Sequence[float]] = None,
            **convnext_block_kwargs: Any
    ) -> None:
        super().__init__()

        if drop_path_rates is None:
            self.blocks = nn.ModuleList([
                ConvNeXtBlock3d(channels, channels, **convnext_block_kwargs)
                for _ in range(depth)
            ])
        else:
            assert len(drop_path_rates) == depth

            self.blocks = nn.ModuleList([
                ConvNeXtBlock3d(channels, channels, drop_path_rate=dp_rate, **convnext_block_kwargs)
                for dp_rate in drop_path_rates
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        return x


class ConvNeXt3d(nn.Module):
    def __init__(self, config: ConvNeXt3dConfig) -> None:
        super().__init__()

        drop_path_rates = torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        drop_path_rates = [dp_rates.tolist() for dp_rates in drop_path_rates]

        self.stem = Stem3d(config)
        self.stages = nn.ModuleList([])
        self.lns = nn.ModuleList([])
        self.down_convs = nn.ModuleList([])
        for c_1, c_2, d, dp_rates in zip(config.channels, config.channels[1:], config.depths, drop_path_rates):
            self.stages.append(ConvNeXtStage3d(c_1, d, dp_rates))
            self.lns.append(LayerNorm3d(c_1))
            self.down_convs.append(nn.Conv3d(c_1, c_2, kernel_size=2, stride=2))
        self.stages.append(ConvNeXtStage3d(config.channels[-1], config.depths[-1], drop_path_rates[-1]))
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.final_ln = nn.LayerNorm(config.channels[-1], eps=1e-6)

        self.config = config
        self.min_input_size = tuple(s * 2 ** len(self.down_convs) for s in _to_tuple(config.stem_stride))

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor] = None) -> ConvNeXt3dOutput:
        if any(image.shape[i] < self.min_input_size[i] for i in [-3, -2, -1]):
            raise ValueError(f"Input's spatial size {x.shape[-3:]} is less than {self.min_input_size}.")

        if mask is not None and mask.dtype != image.dtype:
            raise TypeError("``mask`` must have the same dtype as input image ``x``")

        x = self.stem(image, mask)

        feature_pyramid = []
        for stage, ln, conv in zip(self.stages, self.lns, self.down_convs):
            x = stage(x)
            feature_pyramid.append(x)
            x = ln(x)
            x = conv(x)

        x = self.stages[-1](x)
        feature_pyramid.append(x)

        pooled_features = self.avg_pool(x).squeeze((2, 3, 4))
        pooled_features = self.final_ln(pooled_features)

        return ConvNeXt3dOutput(feature_pyramid, pooled_features)


def _to_tuple(int_or_seq: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(int_or_seq, int):
        return (int_or_seq, int_or_seq, int_or_seq)
    else:
        assert len(int_or_seq) == 3

        return tuple(int_or_seq)
