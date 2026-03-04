from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import torch

from scipy.ndimage import binary_dilation



def _disk_structure(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    mask = x * x + y * y <= radius * radius
    return mask.astype(bool)



def boundary_f1_dilated(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    ignore_value: int = 2,
    dilation_px: int = 2,
) -> Dict[str, float]:
    """
    logits: (B,1,H,W)
    target: (B,H,W) values {0,1,ignore}
    """
    prob = torch.sigmoid(logits.squeeze(1))
    pred = (prob >= float(threshold)).detach().cpu().numpy().astype(bool)

    valid = (target != int(ignore_value)).detach().cpu().numpy().astype(bool)
    gt = (target == 1).detach().cpu().numpy().astype(bool)

    pred = np.logical_and(pred, valid)
    gt = np.logical_and(gt, valid)

    structure = _disk_structure(int(dilation_px))

    tp = 0
    fp = 0
    fn = 0
    for i in range(pred.shape[0]):
        p = pred[i]
        g = gt[i]
        dil_p = binary_dilation(p, structure=structure)
        dil_g = binary_dilation(g, structure=structure)

        tp += int(np.logical_and(p, dil_g).sum())
        fp += int(np.logical_and(p, np.logical_not(dil_g)).sum())
        fn += int(np.logical_and(g, np.logical_not(dil_p)).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "boundary_f1": f1,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_tp": float(tp),
        "boundary_fp": float(fp),
        "boundary_fn": float(fn),
    }



def boundary_metrics_multi_threshold(
    logits: torch.Tensor,
    target: torch.Tensor,
    thresholds: Iterable[float],
    ignore_value: int = 2,
    dilation_px: int = 2,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for t in thresholds:
        m = boundary_f1_dilated(
            logits=logits,
            target=target,
            threshold=float(t),
            ignore_value=ignore_value,
            dilation_px=dilation_px,
        )
        key_prefix = f"boundary@{float(t):.2f}"
        out[f"{key_prefix}_f1"] = m["boundary_f1"]
        out[f"{key_prefix}_precision"] = m["boundary_precision"]
        out[f"{key_prefix}_recall"] = m["boundary_recall"]
    return out
