from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _soft_dice_loss(
    logit: torch.Tensor,
    target01: torch.Tensor,
    valid_f: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    prob = torch.sigmoid(logit)
    prob = prob * valid_f
    target01 = target01 * valid_f

    inter = (prob * target01).sum(dim=(1, 2))
    den = prob.sum(dim=(1, 2)) + target01.sum(dim=(1, 2))
    dice = (2.0 * inter + float(smooth)) / (den + float(smooth))
    return 1.0 - dice.mean()


def _resolve_pos_weight(
    logit: torch.Tensor,
    target01: torch.Tensor,
    valid_f: torch.Tensor,
    pos_weight: float | str | None,
) -> tuple[torch.Tensor | None, float | None]:
    if pos_weight is None:
        return None, None

    if isinstance(pos_weight, str):
        p = pos_weight.strip().lower()
        if p in {"", "none", "off", "false"}:
            return None, None
        if p == "auto":
            pos = (target01 * valid_f).sum()
            neg = ((1.0 - target01) * valid_f).sum()
            if float(pos.detach().item()) <= 0.0:
                return None, None
            auto = torch.clamp(neg / pos.clamp_min(1.0), min=1.0, max=16.0)
            used = float(auto.detach().item())
            return torch.as_tensor(used, dtype=logit.dtype, device=logit.device), used
        used = float(p)
    else:
        used = float(pos_weight)

    if used <= 0:
        return None, None
    return torch.as_tensor(used, dtype=logit.dtype, device=logit.device), used


def boundary_bwbl_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_value: int = 2,
    pos_weight: float | str | None = None,
    focal_gamma: float = 0.0,
    bce_weight: float = 1.0,
    dice_weight: float = 0.0,
    dice_smooth: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    logits: (B,1,H,W)
    target: (B,H,W) values {0,1,ignore}
    """
    logit = logits.squeeze(1)

    valid = (target != ignore_value)
    target01 = (target == 1).float()
    valid_f = valid.float()

    pw_t, pw_used = _resolve_pos_weight(
        logit=logit,
        target01=target01,
        valid_f=valid_f,
        pos_weight=pos_weight,
    )
    raw = F.binary_cross_entropy_with_logits(logit, target01, reduction="none", pos_weight=pw_t)

    gamma = float(focal_gamma)
    if gamma > 0:
        prob = torch.sigmoid(logit)
        p_t = target01 * prob + (1.0 - target01) * (1.0 - prob)
        raw = raw * ((1.0 - p_t).clamp_min(1e-6) ** gamma)

    valid_pixels = valid_f.sum()
    bce = (raw * valid_f).sum() / valid_pixels.clamp_min(1.0)
    dice = _soft_dice_loss(logit, target01, valid_f, smooth=float(dice_smooth))
    loss = float(bce_weight) * bce + float(dice_weight) * dice
    pos = (target01 * valid_f).sum()
    pos_frac = float((pos / valid_pixels.clamp_min(1.0)).detach().item()) if float(valid_pixels.detach().item()) > 0 else 0.0

    info = {
        "boundary_total": float(loss.detach().item()),
        "boundary_bce": float(bce.detach().item()),
        "boundary_dice": float(dice.detach().item()),
        "boundary_pos_weight_used": 1.0 if pw_used is None else float(pw_used),
        "boundary_focal_gamma": float(gamma),
        "boundary_pos_frac": float(pos_frac),
        "boundary_valid_pixels": float(valid_pixels.detach().item()),
    }
    return loss, info
