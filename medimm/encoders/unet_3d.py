from typing import Optional, List, Union, Tuple, Sequence, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import DropPath

from medimm.layers.norm import LayerNorm3d, GlobalResponseNorm3d
from medimm.layers.layer_scale import LayerScale3d


class Stem3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: Union[int, Tuple[int, int, int]],
            padding: int,
    ) -> None:
        super().__init__()

        assert out_channels > 1

        self.conv = nn.Conv3d(in_channels, out_channels - 1, kernel_size, stride, padding)
        self.norm = LayerNorm3d(out_channels - 1)
        self.stride = stride

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            x = self.conv(x)
            x = self.norm(x)
            n, _, h, w, d = x.shape
            mask = torch.ones((n, 1, h, w, d), dtype=x.dtype, device=x.device)
            x = torch.cat([x, mask], dim=1)
            return x
        else:
            mask = mask.unsqueeze(1)
            x = self.conv(x * mask)
            mask = F.max_pool3d(mask, kernel_size=self.stride)
            noise = torch.randn_like(x)
            x = x * mask + noise * (1 - mask)
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
        self.norm_1 = LayerNorm3d(in_channels)
        self.act_1 = nn.GELU()
        self.conv_2 = nn.Conv3d(in_channels, hidden_channels, kernel_size=1)
        self.norm_2 = LayerNorm3d(hidden_channels)
        self.act_2 = nn.GELU()
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
        x = self.norm_1(x)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.norm_2(x)
        x = self.act_2(x)
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


class UNet3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: Sequence[int],
            depths: Sequence[Union[int, Tuple[int, int]]],
            stem_stride: Union[int, Tuple[int, int, int]],
            stem_kernel_size: Optional[Union[int, Tuple[int, int, int]]] = None,
            stem_padding: Union[int, Tuple[int, int, int]] = 0,
            drop_path_rate: float = 0.0,
            final_ln: bool = True,
            final_affine: bool = True,
            final_gelu: bool = False,
            **convnext_block_kwargs: Any
    ) -> None:
        super().__init__()

        assert len(out_channels) == len(depths)

        if isinstance(stem_stride, int):
            stem_stride = (stem_stride, stem_stride, stem_stride)
        stem_stride = tuple(stem_stride)

        if stem_kernel_size is None:
            stem_kernel_size = stem_stride

        left_depths = [d for d, _ in depths[:-1]] + [depths[-1]]
        drop_path_rates = torch.linspace(0, drop_path_rate, sum(left_depths)).split(left_depths)
        drop_path_rates = [dp_rates.tolist() for dp_rates in drop_path_rates]

        self.stem = Stem3d(in_channels, out_channels[0], stem_kernel_size, stem_stride, stem_padding)
        self.left_stages = nn.ModuleList([])
        self.middle_norms = nn.ModuleList([])
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.skip_connections = nn.ModuleList([])
        self.right_stages = nn.ModuleList([])
        self.final_norms = nn.ModuleList([])
        self.final_acts = nn.ModuleList([])
        for c_1, c_2, (d_1, d_2), dp_rates in zip(out_channels, out_channels[1:], depths, drop_path_rates):
            self.left_stages.append(ConvNeXtStage3d(c_1, d_1, dp_rates, **convnext_block_kwargs))
            self.middle_norms.append(LayerNorm3d(c_1))
            self.down_blocks.append(nn.Conv3d(c_1, c_2, kernel_size=2, stride=2))
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv3d(c_2, c_1, kernel_size=1),
                    nn.Upsample(scale_factor=2, mode='trilinear')
                )
            )
            self.skip_connections.append(nn.Conv3d(c_1, c_1, kernel_size=1))
            self.right_stages.append(ConvNeXtStage3d(c_1, d_2, **convnext_block_kwargs))
            self.final_norms.append(LayerNorm3d(c_1, affine=final_affine) if final_ln else nn.Identity())
            self.final_acts.append(nn.GELU() if final_gelu else nn.Identity())

        self.bottom_stage = ConvNeXtStage3d(out_channels[-1], depths[-1], drop_path_rates[-1], **convnext_block_kwargs)
        self.final_norms.append(LayerNorm3d(out_channels[-1], affine=final_affine) if final_ln else nn.Identity())
        self.final_acts.append(nn.GELU() if final_gelu else nn.Identity())

        self.out_channels = out_channels
        self.stem_stride = stem_stride
        self.max_stride = tuple(s * 2 ** len(self.down_blocks) for s in stem_stride)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        if any(x.shape[i] < self.max_stride[i] for i in [-3, -2, -1]):
            raise ValueError(f"Input's spatial size {x.shape[-3:]} is less than {self.max_stride}.")

        if mask is not None and mask.dtype != x.dtype:
            raise TypeError("``mask`` must have the same dtype as input image ``x``")

        # stem
        x = self.stem(x, mask)

        # UNet's down path
        pyramid = []
        for i in range(len(self.down_blocks)):
            x = self.left_stages[i](x)
            x = self.middle_norms[i](x)
            pyramid.append(x)
            x = self.down_blocks[i](x)

        # UNet's bottom layers
        x = self.bottom_stage(x)
        pyramid.append(self.final_acts[-1](self.final_norms[-1](x)))

        # UNet's up path
        for i in reversed(range(len(self.up_blocks))):
            x = self.up_blocks[i](x)
            y = self.skip_connections[i](pyramid[i])
            x = crop_and_pad_to(x, y)
            x = x + y
            x = self.right_stages[i](x)
            pyramid[i] = self.final_acts[i](self.final_norms[i](x))

        return pyramid


def crop_and_pad_to(x: torch.Tensor, other: torch.Tensor, pad_mode: str = 'replicate') -> torch.Tensor:
    assert x.ndim == other.ndim == 5

    # crop
    x = x[(..., *map(slice, other.shape[-3:]))]

    # pad
    pad = []
    for dim in [-1, -2, -3]:
        pad += [0, max(other.shape[dim] - x.shape[dim], 0)]
    x = F.pad(x, pad, mode=pad_mode)

    return x
