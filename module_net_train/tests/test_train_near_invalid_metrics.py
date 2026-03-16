from __future__ import annotations

import logging
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from net_train.hardware import RuntimePlan
from net_train.train.loop import (
    _near_invalid_mask_from_valid_batch,
    train_one_epoch,
    validate_one_epoch,
)


class _OneSampleDataset(Dataset):
    def __init__(self, sample: dict[str, object]) -> None:
        self._sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self._sample


class _TinyTwoHeadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        b, _, h, w = x.shape
        logits = self.bias.view(1, 1, 1, 1).expand(b, 1, h, w)
        return {
            "extent_logits": logits,
            "boundary_logits": logits,
        }


def _cpu_plan() -> RuntimePlan:
    return RuntimePlan(
        device="cpu",
        precision="fp32",
        amp_enabled=False,
        amp_dtype="float32",
        crop_size=256,
        batch_size=1,
        grad_accum_steps=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=2,
        channels_last=False,
        cudnn_benchmark=False,
        allow_tf32=False,
        torch_compile=False,
        gpu_name=None,
        gpu_vram_gb=None,
        warnings=[],
    )


def _base_train_cfg() -> dict[str, object]:
    return {
        "dataset": {
            "targets": {
                "extent": {"ignore_value": 255},
                "boundary_bwbl": {"ignore_value": 2},
            }
        },
        "loss": {
            "weights": {"extent": 1.0, "boundary": 1.0},
            "extent": {"ignore_value": 255, "bce_weight": 1.0, "dice_weight": 1.0},
            "boundary_bwbl": {
                "ignore_value": 2,
                "pos_weight": "auto",
                "focal_gamma": 0.0,
                "bce_weight": 1.0,
                "dice_weight": 0.0,
            },
        },
        "metrics": {
            "extent": {"threshold": 0.5},
            "boundary": {"thresholds": [0.5], "dilation_px": 1},
        },
        "train": {
            "batch": {"grad_accum_steps": 1},
            "log_every_n_steps": 0,
        },
    }


def _make_sample(*, with_invalid: bool, ignore_near_boundary: bool) -> dict[str, object]:
    h = w = 6
    image = torch.zeros((9, h, w), dtype=torch.float32)
    extent = torch.zeros((h, w), dtype=torch.long)
    boundary = torch.zeros((h, w), dtype=torch.long)
    valid = torch.ones((h, w), dtype=torch.float32)

    extent[2:4, 2:4] = 1
    boundary[2:4, 2] = 1
    if with_invalid:
        valid[:, 0] = 0.0
    if ignore_near_boundary:
        boundary[:, 1] = 2

    return {
        "image": image,
        "extent": extent,
        "boundary": boundary,
        "valid": valid,
        "near_invalid_ratio": 0.0,
        "valid_ratio": float(valid.mean().item()),
        "synthetic_invalid_applied": 0.0,
    }


def test_near_invalid_mask_from_valid_batch_marks_valid_side_only() -> None:
    valid = torch.ones((1, 5, 5), dtype=torch.float32)
    valid[0, 2, 2] = 0.0

    near = _near_invalid_mask_from_valid_batch(
        valid,
        radius_px=1,
        out_device=torch.device("cpu"),
    )
    assert near.dtype == torch.bool
    assert tuple(near.shape) == (1, 5, 5)
    assert bool(near[0, 2, 2]) is False  # invalid center is never part of near-valid subset
    assert bool(near[0, 2, 1]) is True
    assert bool(near[0, 1, 2]) is True
    assert bool(near[0, 0, 0]) is False


def test_validate_near_invalid_metrics_empty_band_is_nan() -> None:
    cfg = _base_train_cfg()
    ds = _OneSampleDataset(_make_sample(with_invalid=False, ignore_near_boundary=False))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = _TinyTwoHeadModel()

    metrics = validate_one_epoch(
        model=model,
        loader=loader,
        plan=_cpu_plan(),
        train_cfg=cfg,
    )

    assert metrics["val/near_invalid_valid_frac"] == 0.0
    assert math.isnan(float(metrics["val/extent_f1_near_invalid"]))
    assert math.isnan(float(metrics["val/boundary_f1_max_near_invalid"]))
    assert math.isnan(float(metrics["val/boundary_prob_gap_near_invalid"]))


def test_validate_near_invalid_boundary_metrics_respect_ignore() -> None:
    cfg = _base_train_cfg()
    cfg.setdefault("train", {})
    cfg["train"]["diagnostics"] = {"near_invalid_radius_px": 1}
    ds = _OneSampleDataset(_make_sample(with_invalid=True, ignore_near_boundary=True))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = _TinyTwoHeadModel()

    metrics = validate_one_epoch(
        model=model,
        loader=loader,
        plan=_cpu_plan(),
        train_cfg=cfg,
    )

    assert float(metrics["val/near_invalid_valid_frac"]) > 0.0
    assert math.isfinite(float(metrics["val/extent_f1_near_invalid"]))
    assert math.isnan(float(metrics["val/boundary_f1_max_near_invalid"]))
    assert math.isnan(float(metrics["val/boundary_prob_gap_near_invalid"]))


def test_train_one_epoch_reports_near_invalid_supervision_without_new_config() -> None:
    cfg = _base_train_cfg()  # no explicit train.diagnostics block: backward-compatible path
    ds = _OneSampleDataset(_make_sample(with_invalid=True, ignore_near_boundary=False))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    model = _TinyTwoHeadModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    metrics = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        plan=_cpu_plan(),
        train_cfg=cfg,
        scaler=None,
        epoch=1,
        logger=logging.getLogger("test_train_near_invalid_metrics"),
    )

    assert "train/extent_supervised_frac" in metrics
    assert "train/boundary_supervised_frac" in metrics
    assert "train/boundary_pos_frac_near_invalid" in metrics
    assert "train/near_invalid_supervised_frac" in metrics
    assert 0.0 <= float(metrics["train/extent_supervised_frac"]) <= 1.0
    assert 0.0 <= float(metrics["train/boundary_supervised_frac"]) <= 1.0
    assert math.isfinite(float(metrics["train/boundary_pos_frac_near_invalid"]))
    assert math.isfinite(float(metrics["train/near_invalid_supervised_frac"]))
