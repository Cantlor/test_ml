from __future__ import annotations

from contextlib import nullcontext
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional
import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from net_train.data.transforms import near_invalid_band
from net_train.hardware import RuntimePlan, amp_dtype_from_plan
from net_train.losses.bwbl_loss import boundary_bwbl_loss
from net_train.losses.extent_loss import extent_loss
from net_train.metrics.boundary_metrics import boundary_f1_dilated
from net_train.metrics.extent_metrics import extent_binary_metrics
from net_train.train.checkpoint import CheckpointManager
from net_train.utils.io import append_csv_row
from net_train.utils.progress import iter_progress



def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = dict(batch)
    out["image"] = out["image"].to(device, non_blocking=True)
    out["extent"] = out["extent"].to(device, non_blocking=True)
    out["boundary"] = out["boundary"].to(device, non_blocking=True)
    return out



def _autocast_context(plan: RuntimePlan):
    if plan.device != "cuda" or not plan.amp_enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype_from_plan(plan))



def _optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    grad_clip: float,
) -> None:
    if grad_clip > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _safe_ratio(num: float, den: float, default: float) -> float:
    return float(num / den) if den > 0 else float(default)


def _safe_ratio_nan(num: float, den: float) -> float:
    return _safe_ratio(num, den, float("nan"))


def _f1_from_counts(tp: float, fp: float, fn: float) -> float:
    precision = _safe_ratio(tp, tp + fp, 1.0)
    recall = _safe_ratio(tp, tp + fn, 1.0)
    return _safe_ratio(2.0 * precision * recall, precision + recall, 0.0)


def _f1_from_counts_nan(tp: float, fp: float, fn: float) -> float:
    if (tp + fp + fn) <= 0:
        return float("nan")
    precision = _safe_ratio(tp, tp + fp, 1.0)
    recall = _safe_ratio(tp, tp + fn, 1.0)
    return _safe_ratio(2.0 * precision * recall, precision + recall, 0.0)


def _resolve_near_invalid_radius_px(train_cfg: Dict[str, Any]) -> int:
    train_section = train_cfg.get("train", {}) or {}
    diagnostics_cfg = train_section.get("diagnostics", {}) or {}
    if "near_invalid_radius_px" in diagnostics_cfg:
        return max(0, int(diagnostics_cfg.get("near_invalid_radius_px", 2)))

    sampling_cfg = train_cfg.get("sampling", {}) or {}
    crop_policy_cfg = sampling_cfg.get("crop_policy", {}) or {}
    near_cfg = crop_policy_cfg.get("near_invalid", {}) or {}
    fallback = near_cfg.get("diag_radius_px", near_cfg.get("radius_px", 2))
    return max(0, int(fallback))


def _near_invalid_mask_from_valid_batch(
    valid: torch.Tensor,
    *,
    radius_px: int,
    out_device: torch.device,
) -> torch.Tensor:
    radius = max(0, int(radius_px))
    if radius == 0:
        out = np.zeros(tuple(valid.shape), dtype=bool)
        return torch.from_numpy(out).to(device=out_device, dtype=torch.bool)

    valid_np = (valid.detach().cpu().numpy() > 0.5).astype(np.uint8)
    near_np = np.zeros_like(valid_np, dtype=bool)
    for i in range(valid_np.shape[0]):
        near_np[i] = near_invalid_band(valid_np[i], radius_px=radius)
    return torch.from_numpy(near_np).to(device=out_device, dtype=torch.bool)


def _loss_and_components(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_cfg: Dict[str, Any],
) -> tuple[torch.Tensor, Dict[str, float]]:
    weights = (loss_cfg.get("weights", {}) or {})
    w_extent = float(weights.get("extent", 1.0))
    w_boundary = float(weights.get("boundary", 1.0))

    extent_cfg = (loss_cfg.get("extent", {}) or {})
    boundary_cfg = (loss_cfg.get("boundary_bwbl", {}) or {})

    extent_l, extent_info = extent_loss(
        logits=outputs["extent_logits"],
        target=batch["extent"],
        ignore_value=int(extent_cfg.get("ignore_value", 255)),
        bce_weight=float(extent_cfg.get("bce_weight", 1.0)),
        dice_weight=float(extent_cfg.get("dice_weight", 1.0)),
    )

    boundary_l, boundary_info = boundary_bwbl_loss(
        logits=outputs["boundary_logits"],
        target=batch["boundary"],
        ignore_value=int(boundary_cfg.get("ignore_value", 2)),
        pos_weight=boundary_cfg.get("pos_weight", None),
        focal_gamma=float(boundary_cfg.get("focal_gamma", 0.0)),
        bce_weight=float(boundary_cfg.get("bce_weight", 1.0)),
        dice_weight=float(boundary_cfg.get("dice_weight", 0.0)),
        dice_smooth=float(boundary_cfg.get("dice_smooth", 1.0)),
    )

    total = w_extent * extent_l + w_boundary * boundary_l

    info = {
        "loss": float(total.detach().item()),
        **extent_info,
        **boundary_info,
    }
    return total, info


def _batch_mean_scalar(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().float().mean().item())
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        vals = [float(v) for v in x]
        return float(sum(vals) / max(1, len(vals)))
    return float("nan")



def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    plan: RuntimePlan,
    train_cfg: Dict[str, Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    logger,
    show_progress: bool | None = None,
) -> Dict[str, float]:
    model.train()

    train_section = train_cfg.get("train", {}) or {}
    batch_section = train_section.get("batch", {}) or {}

    grad_clip = float(train_section.get("grad_clip_norm", 0.0))
    grad_accum = int(batch_section.get("grad_accum_steps", 1))
    accum_steps = max(1, grad_accum)
    log_every = int(train_section.get("log_every_n_steps", 20))
    log_smooth_window = int(train_section.get("log_smoothing_window", log_every if log_every > 0 else 20))
    log_smooth_window = max(1, log_smooth_window)

    loss_cfg = train_cfg.get("loss", {}) or {}
    dataset_cfg = train_cfg.get("dataset", {}) or {}
    targets_cfg = dataset_cfg.get("targets", {}) or {}
    extent_ignore = int((targets_cfg.get("extent", {}) or {}).get("ignore_value", 255))
    boundary_ignore = int((targets_cfg.get("boundary_bwbl", {}) or {}).get("ignore_value", 2))
    near_invalid_radius_px = _resolve_near_invalid_radius_px(train_cfg)

    running_loss = 0.0
    running_extent = 0.0
    running_boundary = 0.0
    running_boundary_bce = 0.0
    running_boundary_dice = 0.0
    running_boundary_pos_weight = 0.0
    running_boundary_pos_frac = 0.0
    running_near_invalid_ratio = 0.0
    running_valid_ratio = 0.0
    running_synthetic_invalid_applied = 0.0
    running_extent_supervised_frac = 0.0
    running_boundary_supervised_frac = 0.0
    running_boundary_pos_frac_near_invalid = 0.0
    running_near_invalid_supervised_frac = 0.0
    cnt_boundary_pos_frac_near_invalid = 0
    cnt_near_invalid_supervised_frac = 0
    steps = 0
    recent_loss = deque(maxlen=log_smooth_window)
    recent_extent = deque(maxlen=log_smooth_window)
    recent_boundary = deque(maxlen=log_smooth_window)
    recent_near_invalid = deque(maxlen=log_smooth_window)
    recent_synth = deque(maxlen=log_smooth_window)

    optimizer.zero_grad(set_to_none=True)
    device = torch.device(plan.device)

    batch_iter = iter_progress(
        loader,
        total=len(loader),
        desc=f"train e{epoch}",
        unit="batch",
        enabled=show_progress,
        leave=False,
    )
    for step_idx, batch in enumerate(batch_iter, start=1):
        batch = _to_device(batch, device=device)

        with _autocast_context(plan):
            out = model(batch["image"])
            loss, info = _loss_and_components(out, batch, loss_cfg)
            loss_scaled = loss / accum_steps

        if not torch.isfinite(loss):
            logger.warning(
                f"epoch={epoch} step={step_idx}/{len(loader)} non-finite loss detected; batch skipped"
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if step_idx % accum_steps == 0:
            _optimizer_step(model=model, optimizer=optimizer, scaler=scaler, grad_clip=grad_clip)

        steps += 1
        running_loss += float(info["loss"])
        running_extent += float(info["extent_total"])
        running_boundary += float(info["boundary_total"])
        running_boundary_bce += float(info.get("boundary_bce", info["boundary_total"]))
        running_boundary_dice += float(info.get("boundary_dice", 0.0))
        running_boundary_pos_weight += float(info.get("boundary_pos_weight_used", 1.0))
        running_boundary_pos_frac += float(info.get("boundary_pos_frac", 0.0))
        near_invalid_ratio = _batch_mean_scalar(batch.get("near_invalid_ratio"))
        valid_ratio = _batch_mean_scalar(batch.get("valid_ratio"))
        synth_applied = _batch_mean_scalar(batch.get("synthetic_invalid_applied"))
        extent_supervised = (batch["extent"] != extent_ignore)
        boundary_supervised = (batch["boundary"] != boundary_ignore)
        extent_supervised_frac = float(extent_supervised.float().mean().item())
        boundary_supervised_frac = float(boundary_supervised.float().mean().item())
        running_extent_supervised_frac += extent_supervised_frac
        running_boundary_supervised_frac += boundary_supervised_frac
        if isinstance(batch.get("valid"), torch.Tensor):
            near_mask = _near_invalid_mask_from_valid_batch(
                batch["valid"],
                radius_px=near_invalid_radius_px,
                out_device=batch["boundary"].device,
            )
            near_boundary_supervised = near_mask & boundary_supervised
            near_supervised_den = float(near_mask.sum().item())
            near_supervised_num = float(near_boundary_supervised.sum().item())
            near_invalid_supervised_frac = _safe_ratio_nan(near_supervised_num, near_supervised_den)
            near_pos_num = float(((batch["boundary"] == 1) & near_boundary_supervised).sum().item())
            boundary_pos_frac_near_invalid = _safe_ratio_nan(near_pos_num, near_supervised_num)
        else:
            near_invalid_supervised_frac = float("nan")
            boundary_pos_frac_near_invalid = float("nan")
        running_near_invalid_ratio += 0.0 if not math.isfinite(near_invalid_ratio) else near_invalid_ratio
        running_valid_ratio += 0.0 if not math.isfinite(valid_ratio) else valid_ratio
        running_synthetic_invalid_applied += 0.0 if not math.isfinite(synth_applied) else synth_applied
        if math.isfinite(boundary_pos_frac_near_invalid):
            running_boundary_pos_frac_near_invalid += boundary_pos_frac_near_invalid
            cnt_boundary_pos_frac_near_invalid += 1
        if math.isfinite(near_invalid_supervised_frac):
            running_near_invalid_supervised_frac += near_invalid_supervised_frac
            cnt_near_invalid_supervised_frac += 1
        recent_loss.append(float(info["loss"]))
        recent_extent.append(float(info["extent_total"]))
        recent_boundary.append(float(info["boundary_total"]))
        if math.isfinite(near_invalid_ratio):
            recent_near_invalid.append(float(near_invalid_ratio))
        if math.isfinite(synth_applied):
            recent_synth.append(float(synth_applied))

        if log_every > 0 and step_idx % log_every == 0:
            avg_loss = sum(recent_loss) / max(1, len(recent_loss))
            avg_extent = sum(recent_extent) / max(1, len(recent_extent))
            avg_boundary = sum(recent_boundary) / max(1, len(recent_boundary))
            avg_near = sum(recent_near_invalid) / max(1, len(recent_near_invalid))
            avg_synth = sum(recent_synth) / max(1, len(recent_synth))
            batch_iter.set_postfix(
                loss=f"{avg_loss:.4f}",
                extent=f"{avg_extent:.4f}",
                boundary=f"{avg_boundary:.4f}",
                near=f"{avg_near:.3f}",
                refresh=False,
            )
            logger.info(
                f"epoch={epoch} step={step_idx}/{len(loader)} "
                f"loss={info['loss']:.5f} extent={info['extent_total']:.5f} boundary={info['boundary_total']:.5f} "
                f"| smooth{log_smooth_window}: loss={avg_loss:.5f} extent={avg_extent:.5f} boundary={avg_boundary:.5f} "
                f"near_invalid={avg_near:.3f} synth_applied={avg_synth:.3f}"
            )

    # flush leftover gradients
    if steps % accum_steps != 0:
        _optimizer_step(model=model, optimizer=optimizer, scaler=scaler, grad_clip=grad_clip)

    avg_loss = running_loss / max(1, steps)
    return {
        "train/loss": avg_loss,
        "train/extent_total": running_extent / max(1, steps),
        "train/boundary_total": running_boundary / max(1, steps),
        "train/boundary_bce": running_boundary_bce / max(1, steps),
        "train/boundary_dice": running_boundary_dice / max(1, steps),
        "train/boundary_pos_weight": running_boundary_pos_weight / max(1, steps),
        "train/boundary_pos_frac": running_boundary_pos_frac / max(1, steps),
        "train/near_invalid_ratio": running_near_invalid_ratio / max(1, steps),
        "train/valid_ratio": running_valid_ratio / max(1, steps),
        "train/synthetic_invalid_applied_rate": running_synthetic_invalid_applied / max(1, steps),
        "train/extent_supervised_frac": running_extent_supervised_frac / max(1, steps),
        "train/boundary_supervised_frac": running_boundary_supervised_frac / max(1, steps),
        "train/boundary_pos_frac_near_invalid": _safe_ratio(
            running_boundary_pos_frac_near_invalid,
            float(cnt_boundary_pos_frac_near_invalid),
            float("nan"),
        ),
        "train/near_invalid_supervised_frac": _safe_ratio(
            running_near_invalid_supervised_frac,
            float(cnt_near_invalid_supervised_frac),
            float("nan"),
        ),
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    plan: RuntimePlan,
    train_cfg: Dict[str, Any],
    epoch: int | None = None,
    show_progress: bool | None = None,
) -> Dict[str, float]:
    model.eval()

    loss_cfg = train_cfg.get("loss", {}) or {}
    extent_cfg = ((train_cfg.get("metrics", {}) or {}).get("extent", {}) or {})
    boundary_cfg = ((train_cfg.get("metrics", {}) or {}).get("boundary", {}) or {})

    extent_threshold = float(extent_cfg.get("threshold", 0.5))
    boundary_thresholds = [float(v) for v in boundary_cfg.get("thresholds", [0.5, 0.35])]
    if not boundary_thresholds:
        raise RuntimeError("metrics.boundary.thresholds must contain at least one value")
    boundary_dilation = int(boundary_cfg.get("dilation_px", 2))
    near_invalid_radius_px = _resolve_near_invalid_radius_px(train_cfg)

    dataset_cfg = train_cfg.get("dataset", {}) or {}
    targets_cfg = dataset_cfg.get("targets", {}) or {}
    extent_ignore = int((targets_cfg.get("extent", {}) or {}).get("ignore_value", 255))
    boundary_ignore = int((targets_cfg.get("boundary_bwbl", {}) or {}).get("ignore_value", 2))

    loss_sum = 0.0
    steps = 0

    # aggregate confusion terms for stable epoch-level metrics
    tp_e = fp_e = fn_e = 0.0

    boundary_counts: Dict[float, Dict[str, float]] = {
        t: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for t in boundary_thresholds
    }
    near_boundary_counts: Dict[float, Dict[str, float]] = {
        t: {"tp": 0.0, "fp": 0.0, "fn": 0.0} for t in boundary_thresholds
    }
    tp_e_near = fp_e_near = fn_e_near = 0.0
    near_valid_pixels_total = 0.0
    extent_valid_pixels_total = 0.0
    sum_boundary_prob_pos = 0.0
    cnt_boundary_pos = 0.0
    sum_boundary_prob_neg = 0.0
    cnt_boundary_neg = 0.0
    sum_boundary_prob_pos_near = 0.0
    cnt_boundary_pos_near = 0.0
    sum_boundary_prob_neg_near = 0.0
    cnt_boundary_neg_near = 0.0
    device = torch.device(plan.device)

    val_desc = f"val e{epoch}" if epoch is not None else "validation"
    batch_iter = iter_progress(
        loader,
        total=len(loader),
        desc=val_desc,
        unit="batch",
        enabled=show_progress,
        leave=False,
    )
    for batch in batch_iter:
        batch = _to_device(batch, device=device)
        near_mask = None
        if isinstance(batch.get("valid"), torch.Tensor):
            near_mask = _near_invalid_mask_from_valid_batch(
                batch["valid"],
                radius_px=near_invalid_radius_px,
                out_device=batch["extent"].device,
            )

        with _autocast_context(plan):
            out = model(batch["image"])
            loss, _ = _loss_and_components(out, batch, loss_cfg)

        loss_sum += float(loss.detach().item())
        steps += 1

        m_extent = extent_binary_metrics(
            logits=out["extent_logits"],
            target=batch["extent"],
            threshold=extent_threshold,
            ignore_value=extent_ignore,
        )
        tp_e += m_extent["extent_tp"]
        fp_e += m_extent["extent_fp"]
        fn_e += m_extent["extent_fn"]
        extent_valid_pixels_total += float((batch["extent"] != extent_ignore).sum().item())

        prob_boundary = torch.sigmoid(out["boundary_logits"].squeeze(1))
        valid_boundary = batch["boundary"] != boundary_ignore
        pos_boundary = (batch["boundary"] == 1) & valid_boundary
        neg_boundary = (batch["boundary"] == 0) & valid_boundary
        if torch.any(pos_boundary):
            sum_boundary_prob_pos += float(prob_boundary[pos_boundary].sum().item())
            cnt_boundary_pos += float(pos_boundary.sum().item())
        if torch.any(neg_boundary):
            sum_boundary_prob_neg += float(prob_boundary[neg_boundary].sum().item())
            cnt_boundary_neg += float(neg_boundary.sum().item())

        if near_mask is not None:
            extent_pred = torch.sigmoid(out["extent_logits"].squeeze(1)) >= float(extent_threshold)
            extent_gt = batch["extent"] == 1
            extent_valid = batch["extent"] != extent_ignore
            near_extent = near_mask & extent_valid
            near_valid_pixels_total += float(near_extent.sum().item())
            tp_e_near += float((extent_pred & extent_gt & near_extent).sum().item())
            fp_e_near += float((extent_pred & (~extent_gt) & near_extent).sum().item())
            fn_e_near += float(((~extent_pred) & extent_gt & near_extent).sum().item())

            near_boundary_valid = near_mask & valid_boundary
            target_boundary_near = batch["boundary"].clone()
            target_boundary_near[~near_boundary_valid] = boundary_ignore
            pos_boundary_near = (batch["boundary"] == 1) & near_boundary_valid
            neg_boundary_near = (batch["boundary"] == 0) & near_boundary_valid
            if torch.any(pos_boundary_near):
                sum_boundary_prob_pos_near += float(prob_boundary[pos_boundary_near].sum().item())
                cnt_boundary_pos_near += float(pos_boundary_near.sum().item())
            if torch.any(neg_boundary_near):
                sum_boundary_prob_neg_near += float(prob_boundary[neg_boundary_near].sum().item())
                cnt_boundary_neg_near += float(neg_boundary_near.sum().item())
        else:
            target_boundary_near = None

        for t in boundary_thresholds:
            m_boundary = boundary_f1_dilated(
                logits=out["boundary_logits"],
                target=batch["boundary"],
                threshold=float(t),
                ignore_value=boundary_ignore,
                dilation_px=boundary_dilation,
            )
            boundary_counts[t]["tp"] += float(m_boundary["boundary_tp"])
            boundary_counts[t]["fp"] += float(m_boundary["boundary_fp"])
            boundary_counts[t]["fn"] += float(m_boundary["boundary_fn"])
            if target_boundary_near is not None:
                m_boundary_near = boundary_f1_dilated(
                    logits=out["boundary_logits"],
                    target=target_boundary_near,
                    threshold=float(t),
                    ignore_value=boundary_ignore,
                    dilation_px=boundary_dilation,
                )
                near_boundary_counts[t]["tp"] += float(m_boundary_near["boundary_tp"])
                near_boundary_counts[t]["fp"] += float(m_boundary_near["boundary_fp"])
                near_boundary_counts[t]["fn"] += float(m_boundary_near["boundary_fn"])

    iou = _safe_ratio(tp_e, tp_e + fp_e + fn_e, 1.0)
    f1 = _f1_from_counts(tp_e, fp_e, fn_e)

    out_metrics: Dict[str, float] = {
        "val/loss": float(loss_sum / max(1, steps)),
        "val/extent_iou": iou,
        "val/extent_f1": f1,
    }

    for t in boundary_thresholds:
        tp_b = float(boundary_counts[t]["tp"])
        fp_b = float(boundary_counts[t]["fp"])
        fn_b = float(boundary_counts[t]["fn"])
        precision_b = _safe_ratio(tp_b, tp_b + fp_b, 1.0)
        recall_b = _safe_ratio(tp_b, tp_b + fn_b, 1.0)
        out_metrics[f"val/boundary_f1@{t:.2f}"] = _f1_from_counts(tp_b, fp_b, fn_b)
        out_metrics[f"val/boundary_precision@{t:.2f}"] = precision_b
        out_metrics[f"val/boundary_recall@{t:.2f}"] = recall_b
    if boundary_thresholds:
        best_t = max(boundary_thresholds, key=lambda t: out_metrics[f"val/boundary_f1@{t:.2f}"])
        out_metrics["val/boundary_f1_max"] = float(out_metrics[f"val/boundary_f1@{best_t:.2f}"])
        out_metrics["val/boundary_f1_max_threshold"] = float(best_t)
        out_metrics["val/boundary_precision_max"] = float(out_metrics[f"val/boundary_precision@{best_t:.2f}"])
        out_metrics["val/boundary_recall_max"] = float(out_metrics[f"val/boundary_recall@{best_t:.2f}"])

    pos_mean = _safe_ratio(sum_boundary_prob_pos, cnt_boundary_pos, 0.0)
    neg_mean = _safe_ratio(sum_boundary_prob_neg, cnt_boundary_neg, 0.0)
    out_metrics["val/boundary_prob_pos_mean"] = pos_mean
    out_metrics["val/boundary_prob_neg_mean"] = neg_mean
    out_metrics["val/boundary_prob_gap"] = float(pos_mean - neg_mean)

    out_metrics["val/near_invalid_valid_frac"] = _safe_ratio(
        near_valid_pixels_total,
        extent_valid_pixels_total,
        0.0,
    )

    extent_iou_near = _safe_ratio_nan(tp_e_near, tp_e_near + fp_e_near + fn_e_near)
    extent_f1_near = _f1_from_counts_nan(tp_e_near, fp_e_near, fn_e_near)
    out_metrics["val/extent_iou_near_invalid"] = extent_iou_near
    out_metrics["val/extent_f1_near_invalid"] = extent_f1_near

    if near_valid_pixels_total > 0:
        best_t_near = None
        best_f1_near = float("-inf")
        for t in boundary_thresholds:
            tp_b = float(near_boundary_counts[t]["tp"])
            fp_b = float(near_boundary_counts[t]["fp"])
            fn_b = float(near_boundary_counts[t]["fn"])
            f1_b = _f1_from_counts_nan(tp_b, fp_b, fn_b)
            if math.isfinite(f1_b) and f1_b > best_f1_near:
                best_f1_near = f1_b
                best_t_near = float(t)
        out_metrics["val/boundary_f1_max_near_invalid"] = (
            float(best_f1_near) if best_t_near is not None else float("nan")
        )
        out_metrics["val/boundary_best_thr_near_invalid"] = (
            float(best_t_near) if best_t_near is not None else float("nan")
        )
    else:
        out_metrics["val/boundary_f1_max_near_invalid"] = float("nan")
        out_metrics["val/boundary_best_thr_near_invalid"] = float("nan")

    pos_mean_near = _safe_ratio_nan(sum_boundary_prob_pos_near, cnt_boundary_pos_near)
    neg_mean_near = _safe_ratio_nan(sum_boundary_prob_neg_near, cnt_boundary_neg_near)
    out_metrics["val/boundary_prob_gap_near_invalid"] = (
        float(pos_mean_near - neg_mean_near)
        if math.isfinite(pos_mean_near) and math.isfinite(neg_mean_near)
        else float("nan")
    )

    return out_metrics



def run_training(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    plan: RuntimePlan,
    train_cfg: Dict[str, Any],
    ckpt_manager: CheckpointManager,
    history_csv_path: Path,
    logger,
    show_progress: bool | None = None,
) -> List[Dict[str, float]]:
    train_section = train_cfg.get("train", {}) or {}
    epochs = int(train_section.get("epochs", 1))
    val_every = max(1, int(train_section.get("val_every_epochs", 1)))
    boundary_cfg = ((train_cfg.get("metrics", {}) or {}).get("boundary", {}) or {})
    boundary_thresholds = [float(v) for v in boundary_cfg.get("thresholds", [0.5, 0.35])]
    skipped_val_template: Dict[str, float] = {
        "val/loss": float("nan"),
        "val/extent_iou": float("nan"),
        "val/extent_f1": float("nan"),
        "val/extent_iou_near_invalid": float("nan"),
        "val/extent_f1_near_invalid": float("nan"),
        "val/boundary_f1_max": float("nan"),
        "val/boundary_f1_max_threshold": float("nan"),
        "val/boundary_f1_max_near_invalid": float("nan"),
        "val/boundary_best_thr_near_invalid": float("nan"),
        "val/boundary_precision_max": float("nan"),
        "val/boundary_recall_max": float("nan"),
        "val/boundary_prob_pos_mean": float("nan"),
        "val/boundary_prob_neg_mean": float("nan"),
        "val/boundary_prob_gap": float("nan"),
        "val/boundary_prob_gap_near_invalid": float("nan"),
        "val/near_invalid_valid_frac": float("nan"),
    }
    for t in boundary_thresholds:
        skipped_val_template[f"val/boundary_f1@{t:.2f}"] = float("nan")
        skipped_val_template[f"val/boundary_precision@{t:.2f}"] = float("nan")
        skipped_val_template[f"val/boundary_recall@{t:.2f}"] = float("nan")

    use_scaler = plan.device == "cuda" and plan.amp_enabled and plan.amp_dtype == "float16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    history: List[Dict[str, float]] = []

    epoch_iter = iter_progress(
        range(1, epochs + 1),
        total=epochs,
        desc="training",
        unit="epoch",
        enabled=show_progress,
    )
    for epoch in epoch_iter:
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            plan=plan,
            train_cfg=train_cfg,
            scaler=scaler if use_scaler else None,
            epoch=epoch,
            logger=logger,
            show_progress=show_progress,
        )

        should_validate = (epoch % val_every == 0) or (epoch == epochs)
        if should_validate:
            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                plan=plan,
                train_cfg=train_cfg,
                epoch=epoch,
                show_progress=show_progress,
            )
        else:
            val_metrics = dict(skipped_val_template)

        if scheduler is not None:
            scheduler.step()

        row: Dict[str, float] = {"epoch": float(epoch)}
        row.update(train_metrics)
        row.update(val_metrics)
        row["lr"] = float(optimizer.param_groups[0]["lr"])
        if ckpt_manager.monitor not in row:
            row[ckpt_manager.monitor] = float("nan")

        history.append(row)
        append_csv_row(history_csv_path, row)

        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if use_scaler else None,
            "metrics": row,
            "runtime_plan": plan.to_dict(),
        }

        ckpt_info = ckpt_manager.step(state=state, metrics=row)
        epoch_iter.set_postfix(
            train=f"{row['train/loss']:.4f}",
            val=f"{row.get('val/loss', float('nan')):.4f}",
            bf1=f"{row.get('val/boundary_f1_max', float('nan')):.4f}",
            refresh=False,
        )

        logger.info(
            f"epoch={epoch}/{epochs} "
            f"train_loss={row['train/loss']:.5f} "
            f"val_loss={row.get('val/loss', float('nan')):.5f} "
            f"val_extent_iou={row.get('val/extent_iou', float('nan')):.4f} "
            f"val_boundary_f1_max={row.get('val/boundary_f1_max', float('nan')):.4f} "
            f"@thr={row.get('val/boundary_f1_max_threshold', float('nan')):.2f} "
            f"monitor={ckpt_info['monitor']}:{ckpt_info['monitor_value']:.5f} "
            f"best={ckpt_info['best_value']}"
        )

    return history
