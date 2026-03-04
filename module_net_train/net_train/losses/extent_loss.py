from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F



def soft_dice_loss(
    logits: torch.Tensor,
    target01: torch.Tensor,
    valid_mask: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)

    probs = probs * valid_mask
    target01 = target01 * valid_mask

    dims = (1, 2)
    inter = (probs * target01).sum(dim=dims)
    den = probs.sum(dim=dims) + target01.sum(dim=dims)
    dice = (2.0 * inter + smooth) / (den + smooth)
    return 1.0 - dice.mean()



def extent_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_value: int = 255,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    logits: (B,1,H,W)
    target: (B,H,W) values {0,1,ignore}
    """
    logit = logits.squeeze(1)

    valid = (target != ignore_value)
    target01 = (target == 1).float()

    bce_raw = F.binary_cross_entropy_with_logits(logit, target01, reduction="none")
    valid_f = valid.float()
    bce = (bce_raw * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    dice = soft_dice_loss(logit, target01, valid_f)

    total = float(bce_weight) * bce + float(dice_weight) * dice

    info = {
        "extent_bce": float(bce.detach().item()),
        "extent_dice": float(dice.detach().item()),
        "extent_total": float(total.detach().item()),
    }
    return total, info
