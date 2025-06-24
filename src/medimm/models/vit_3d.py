from typing import NamedTuple, Tuple, List, Optional

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from timm.layers.helpers import to_3tuple


class ViT3dConfig(NamedTuple):
    image_size: Tuple[int, int, int] = (160, 160, 160)
    patch_size: int = 16
    in_channels: int = 1
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    num_registers: int = 4
    drop_path_rate: float = 0.0
    layer_scale: bool = False
    patch_pos_embed_only: bool = False


class ViT3dOutput(NamedTuple):
    cls_token: torch.Tensor
    reg_tokens: Optional[torch.Tensor]
    patch_tokens: torch.Tensor
    intermediate_cls_tokens: List[torch.Tensor]
    intermediate_reg_tokens: Optional[List[torch.Tensor]]
    intermediate_patch_tokens: List[torch.Tensor]


class PatchEmbed3d(nn.Module):
    def __init__(self, config: ViT3dConfig) -> None:
        super().__init__()

        patch_size = to_3tuple(config.patch_size)
        self.proj = nn.Conv3d(config.in_channels, config.embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = self.proj(image)
        x = x.flatten(2).movedim(-1, 1)
        return x


class ViT3d(nn.Module):
    def __init__(self, config: ViT3dConfig) -> None:
        super().__init__()

        self.config = config

        image_size = to_3tuple(config.image_size)
        patch_size = to_3tuple(config.patch_size)

        if not all(s % p == 0 for s, p in zip(image_size, patch_size)):
            raise ValueError('Image size must be divisible by patch size')

        grid_size = tuple(s // p for s, p in zip(image_size, patch_size))
        num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        num_pos_embeds = 1 + config.num_registers + num_patches if not config.patch_pos_embed_only else num_patches

        self.patch_embed = PatchEmbed3d(config)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.reg_token = nn.Parameter(torch.randn(1, config.num_registers, config.embed_dim) * 0.02) if config.num_registers > 0 else None
        self.pos_embed = nn.Parameter(torch.randn(1, num_pos_embeds, config.embed_dim) * 0.02)

        drop_path_rates = torch.linspace(0, config.drop_path_rate, config.depth).tolist()
        self.blocks = nn.Sequential(*[
            Block(
                dim=config.embed_dim,
                num_heads=config.num_heads,
                qkv_bias=True,
                init_values=1e-5 if config.layer_scale else None,
                drop_path=drop_path_rates[i],
            )
            for i in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)

        self._num_patches = num_patches
        self._grid_size = grid_size

    def forward(self, image: torch.Tensor) -> ViT3dOutput:
        batch_size = image.size(0)

        x = self.patch_embed(image)

        to_cat = []
        to_cat.append(self.cls_token.expand(batch_size, -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(batch_size, -1, -1))

        if self.config.patch_pos_embed_only:
            x = x + self.pos_embed
            x = torch.cat(to_cat + [x], dim=1)
        else:
            x = torch.cat(to_cat + [x], dim=1)
            x = x + self.pos_embed

        intermediate_cls_tokens, intermediate_reg_tokens, intermediate_patch_tokens = [], [], []
        for block in self.blocks:
            x = block(x)

            cls_token, reg_tokens, patch_tokens = x.split((1, self.config.num_registers, self._num_patches), dim=1)
            cls_token = cls_token.squeeze(1)
            patch_tokens = patch_tokens.movedim(1, -1).view(batch_size, self.config.embed_dim, *self._grid_size)

            intermediate_cls_tokens.append(cls_token)
            intermediate_reg_tokens.append(reg_tokens)
            intermediate_patch_tokens.append(patch_tokens)

        x = self.norm(x)

        cls_token, reg_tokens, patch_tokens = x.split((1, self.config.num_registers, self._num_patches), dim=1)
        cls_token = cls_token.squeeze(1)
        patch_tokens = patch_tokens.movedim(1, -1).view(batch_size, self.config.embed_dim, *self._grid_size)

        if self.reg_token is None:
            reg_tokens = None
            intermediate_reg_tokens = None

        return ViT3dOutput(cls_token, reg_tokens, patch_tokens, intermediate_cls_tokens,
                           intermediate_reg_tokens, intermediate_patch_tokens)


def vit3d_base_patch16_reg4_160_dinov2(freeze_pretrained_params: bool = False) -> ViT3d:
    from timm.models.vision_transformer import vit_base_patch14_reg4_dinov2

    config = ViT3dConfig(
        image_size=(160, 160, 160),
        patch_size=16,
        in_channels=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_registers=4,
        drop_path_rate=0,
        layer_scale=True,
        patch_pos_embed_only=True
    )
    model = ViT3d(config)

    pretrained_model = vit_base_patch14_reg4_dinov2(pretrained=True)
    state_dict = pretrained_model.state_dict()
    state_dict.pop('patch_embed.proj.weight')
    state_dict.pop('patch_embed.proj.bias')
    state_dict.pop('pos_embed')

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
    assert unexpected_keys == []

    if freeze_pretrained_params:
        for name, param in model.named_parameters():
            if name not in missing_keys:
                param.requires_grad = False

    return model


def vit3d_large_patch16_reg4_160_dinov2(freeze_pretrained_params: bool = False) -> ViT3d:
    from timm.models.vision_transformer import vit_large_patch14_reg4_dinov2

    config = ViT3dConfig(
        image_size=(160, 160, 160),
        patch_size=16,
        in_channels=1,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        num_registers=4,
        drop_path_rate=0,
        layer_scale=True,
        patch_pos_embed_only=True
    )
    model = ViT3d(config)

    pretrained_model = vit_large_patch14_reg4_dinov2(pretrained=True)
    state_dict = pretrained_model.state_dict()
    state_dict.pop('patch_embed.proj.weight')
    state_dict.pop('patch_embed.proj.bias')
    state_dict.pop('pos_embed')

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert missing_keys == ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']
    assert unexpected_keys == []

    if freeze_pretrained_params:
        for name, param in model.named_parameters():
            if name not in missing_keys:
                param.requires_grad = False

    return model
