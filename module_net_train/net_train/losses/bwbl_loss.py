from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F



def boundary_bwbl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_value: int = 2,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    logits: (B,1,H,W)
    target: (B,H,W) values {0,1,ignore}
    """
    logit = logits.squeeze(1)

    valid = (target != ignore_value)
    target01 = (target == 1).float()

    raw = F.binary_cross_entropy_with_logits(logit, target01, reduction="none")
    valid_f = valid.float()
    loss = (raw * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    info = {
        "boundary_total": float(loss.detach().item()),
    }
    return loss, info
