from typing import NamedTuple, Tuple, List

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from timm.layers.helpers import to_3tuple


class ViT3dConfig(NamedTuple):
    image_size: Tuple[int, int, int] = (128, 128, 128)
    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    num_registers: int = 4
    drop_path_rate: float = 0.0


class ViT3dOutput(NamedTuple):
    cls_token: torch.Tensor
    reg_tokens: torch.Tensor
    patch_tokens: torch.Tensor
    intermediate_cls_tokens: List[torch.Tensor]
    intermediate_reg_tokens: List[torch.Tensor]
    intermediate_patch_tokens: List[torch.Tensor]


class PatchEmbed3d(nn.Module):
    def __init__(self, config: ViT3dConfig) -> None:
        super().__init__()

        patch_size = to_3tuple(config.patch_size)

        self.conv = nn.Conv3d(config.in_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(config.embed_dim)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.conv(image)  # (N, D, H, W, S)
        x = x.flatten(2).movedim(-1, 1)  # (N, H * W * S, D)
        x = self.ln(x)  # (N, H * W * S, D)
        return x


class ViT3d(nn.Module):
    def __init__(self, config: ViT3dConfig) -> None:
        super().__init__()

        self.config = config

        self.image_size = to_3tuple(config.image_size)
        self.patch_size = to_3tuple(config.patch_size)
        self.grid_size = tuple(s // p for s, p in zip(self.image_size, self.patch_size))
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.num_tokens = 1 + config.num_registers + self.num_patches

        self.patch_embed = PatchEmbed3d(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.reg_token = nn.Parameter(torch.zeros(1, config.num_registers, config.embed_dim)) if config.num_registers > 0 else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, config.embed_dim))
        self.norm_pre = nn.LayerNorm(config.embed_dim)

        drop_path_rates = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.blocks = nn.Sequential(*[
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                drop_path=drop_path_rates[i],
            )
            for i in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(image)
        batch_size = x.size(0)

        to_cat = []
        to_cat.append(self.cls_token.expand(batch_size, -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(batch_size, -1, -1))
        x = torch.cat(to_cat + [x], dim=1)

        x = x + self.pos_embed

        x = self.norm_pre(x)

        intermediate_cls_tokens, intermediate_reg_tokens, intermediate_patch_tokens = [], [], []
        for block in self.blocks:
            x = block(x)

            cls_token, reg_tokens, patch_tokens = x.split((1, self.config.num_registers, self.num_patches), dim=1)
            patch_tokens = patch_tokens.view(batch_size, *self.grid_size, self.config.embed_dim)

            intermediate_cls_tokens.append(cls_token)
            intermediate_reg_tokens.append(reg_tokens)
            intermediate_patch_tokens.append(patch_tokens)

        x = self.norm(x)

        cls_token, reg_tokens, patch_tokens = x.split((1, self.config.num_registers, self.num_patches), dim=1)
        patch_tokens = patch_tokens.view(batch_size, *self.grid_size, self.config.embed_dim)

        return ViT3dOutput(cls_token, reg_tokens, patch_tokens, intermediate_cls_tokens,
                           intermediate_reg_tokens, intermediate_patch_tokens)
