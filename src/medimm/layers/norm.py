import torch
import torch.nn as nn


class LayerNorm3d(nn.LayerNorm):
    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__(channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 4, 1)
        x = super().forward(x)
        x = x.permute(0, 4, 1, 2, 3)
        return x


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, channels: int, eps: float = 1e-6, affine: bool = True):
        super().__init__(channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class GlobalResponseNorm3d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_g = x.norm(p=2, dim=(2, 3, 4), keepdim=True)
        x_n = x_g / (x_g.mean(dim=1, keepdim=True) + self.eps)
        x = x + torch.addcmul(self.bias.view(-1, 1, 1, 1), self.weight.view(-1, 1, 1, 1), x * x_n)
        return x
