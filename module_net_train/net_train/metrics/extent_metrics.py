from __future__ import annotations

from typing import Dict

import torch



def extent_binary_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    ignore_value: int = 255,
) -> Dict[str, float]:
    """
    logits: (B,1,H,W)
    target: (B,H,W), values {0,1,ignore}
    """
    prob = torch.sigmoid(logits.squeeze(1))
    pred = prob >= float(threshold)

    valid = target != int(ignore_value)
    gt = target == 1

    pred = pred & valid
    gt = gt & valid

    tp = torch.logical_and(pred, gt).sum().item()
    fp = torch.logical_and(pred, ~gt).sum().item()
    fn = torch.logical_and(~pred, gt).sum().item()

    denom_iou = tp + fp + fn
    iou = float(tp / denom_iou) if denom_iou > 0 else 1.0

    denom_p = tp + fp
    precision = float(tp / denom_p) if denom_p > 0 else 1.0

    denom_r = tp + fn
    recall = float(tp / denom_r) if denom_r > 0 else 1.0

    denom_f1 = precision + recall
    f1 = float(2 * precision * recall / denom_f1) if denom_f1 > 0 else 0.0

    return {
        "extent_iou": iou,
        "extent_f1": f1,
        "extent_precision": precision,
        "extent_recall": recall,
        "extent_tp": float(tp),
        "extent_fp": float(fp),
        "extent_fn": float(fn),
    }
