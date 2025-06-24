from typing import Optional, Literal

import torch


def dice_loss(
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: Literal['mean', 'none'] = 'mean'
) -> torch.Tensor:
    if mask is not None:
        input = torch.where(mask, input, torch.zeros_like(input))
        target = torch.where(mask, target, torch.zeros_like(target))

    dim = tuple(range(2, input.ndim))
    intersection = torch.sum(input * target, dim=dim)
    volumes_sum = torch.sum(input ** 2 + target ** 2, dim=dim)
    loss = 1 - 2 * intersection / (volumes_sum + 1)
    if reduction == 'mean':
        return torch.mean(loss)
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(reduction)
