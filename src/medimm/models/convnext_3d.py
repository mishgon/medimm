from typing import Optional, List, Union, Tuple, Sequence, Any, NamedTuple, Literal

import torch
import torch.nn as nn

from timm.layers import DropPath
from timm.layers.helpers import to_3tuple

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
            drop_path_rates = [0.0] * depth

        assert len(drop_path_rates) == depth

        self.blocks = nn.Sequential(*[
            ConvNeXtBlock3d(channels, channels, drop_path_rate=drop_path_rates[i], **convnext_block_kwargs)
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class ConvNeXt3d(nn.Module):
    def __init__(self, config: ConvNeXt3dConfig) -> None:
        super().__init__()

        self.config = config

        stem_stride = to_3tuple(config.stem_stride)
        if config.stem_kernel_size is not None:
            stem_kernel_size = to_3tuple(config.stem_kernel_size)
        else:
            stem_kernel_size = stem_stride
        stem_padding = to_3tuple(config.stem_padding)

        drop_path_rates = torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)
        drop_path_rates = [dp_rates.tolist() for dp_rates in drop_path_rates]

        self.stem_conv = nn.Conv3d(config.in_channels, config.channels[0], stem_kernel_size, stem_stride, stem_padding)
        self.stem_norm = LayerNorm3d(config.channels[0])
        self.stages = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        self.down_convs = nn.ModuleList([])
        for c_1, c_2, d, dp_rates in zip(config.channels, config.channels[1:], config.depths, drop_path_rates):
            self.stages.append(ConvNeXtStage3d(c_1, d, dp_rates))
            self.norms.append(LayerNorm3d(c_1))
            self.down_convs.append(nn.Conv3d(c_1, c_2, kernel_size=2, stride=2))
        self.stages.append(ConvNeXtStage3d(config.channels[-1], config.depths[-1], drop_path_rates[-1]))
        self.avg_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.final_norm = nn.LayerNorm(config.channels[-1], eps=1e-6)

        self.min_input_size = tuple(s * 2 ** len(self.down_convs) for s in stem_stride)

    def forward(self, image: torch.Tensor) -> ConvNeXt3dOutput:
        if any(image.shape[i] < self.min_input_size[i] for i in [-3, -2, -1]):
            raise ValueError(f"Input's spatial size {x.shape[-3:]} is less than {self.min_input_size}.")

        x = self.stem_conv(image)
        x = self.stem_norm(x)

        feature_pyramid = []
        for stage, norm, conv in zip(self.stages, self.norms, self.down_convs):
            x = stage(x)
            feature_pyramid.append(x)
            x = norm(x)
            x = conv(x)

        x = self.stages[-1](x)
        feature_pyramid.append(x)

        pooled_features = self.avg_pool(x).squeeze((2, 3, 4))
        pooled_features = self.final_norm(pooled_features)

        return ConvNeXt3dOutput(feature_pyramid, pooled_features)
