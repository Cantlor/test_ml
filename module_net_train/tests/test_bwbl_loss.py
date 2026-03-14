from __future__ import annotations

import torch

from net_train.losses.bwbl_loss import boundary_bwbl_loss


def test_boundary_bwbl_loss_auto_pos_weight_runs() -> None:
    logits = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 4, 4), dtype=torch.long)
    target[0, 1, 1] = 1
    target[0, 0, 0] = 2  # ignore

    loss, info = boundary_bwbl_loss(logits, target, ignore_value=2, pos_weight="auto")
    assert torch.isfinite(loss)
    assert float(info["boundary_total"]) >= 0.0
    assert float(info["boundary_pos_weight_used"]) >= 1.0


def test_boundary_bwbl_loss_all_ignore_is_zero() -> None:
    logits = torch.randn((2, 1, 8, 8), dtype=torch.float32)
    target = torch.full((2, 8, 8), 2, dtype=torch.long)

    loss, info = boundary_bwbl_loss(logits, target, ignore_value=2, pos_weight="auto")
    assert float(loss.detach().item()) == 0.0
    assert float(info["boundary_total"]) == 0.0


def test_boundary_bwbl_loss_with_dice_component_runs() -> None:
    logits = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
    target = torch.zeros((1, 6, 6), dtype=torch.long)
    target[0, 2:4, 2:4] = 1

    loss, info = boundary_bwbl_loss(
        logits,
        target,
        ignore_value=2,
        pos_weight="auto",
        focal_gamma=1.0,
        bce_weight=1.0,
        dice_weight=0.25,
    )
    assert torch.isfinite(loss)
    assert float(info["boundary_bce"]) >= 0.0
    assert float(info["boundary_dice"]) >= 0.0
    assert 0.0 <= float(info["boundary_pos_frac"]) <= 1.0
