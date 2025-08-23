import torch
import torch.nn as nn


class LayerScale3d(nn.Module):
    def __init__(self, channels: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()

        self.inplace = inplace
        self.gamma = nn.Parameter(torch.zeros(channels, 1, 1, 1))
        nn.init.constant_(self.gamma, init_values)

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerScale2d(nn.Module):
    def __init__(self, channels: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()

        self.inplace = inplace
        self.gamma = nn.Parameter(torch.zeros(channels, 1, 1))
        nn.init.constant_(self.gamma, init_values)

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

