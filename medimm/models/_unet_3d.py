from typing import NamedTuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext_3d import ConvNeXt3d


class UNet3dConfig(NamedTuple):
    in_channels: int
    out_channels: int
    channels: Sequence[int]
    depths: Sequence[int]


class UNet3d(nn.Module):
    pass


class UNet3dWithConvNextBackbone:
    """After UNet3d encoder, enrich feature pyramid with ConvNext features.
    """
    pass


class UNet3dWithViTBackbone:
    """After UNet3d encoder, enrich feature pyramid with ViT features.
    """
    pass
