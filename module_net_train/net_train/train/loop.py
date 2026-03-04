from __future__ import annotations

from contextlib import nullcontext
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from net_train.hardware import RuntimePlan, amp_dtype_from_plan
from net_train.losses.bwbl_loss import boundary_bwbl_loss
from net_train.losses.extent_loss import extent_loss
from net_train.metrics.boundary_metrics import boundary_metrics_multi_threshold
from net_train.metrics.extent_metrics import extent_binary_metrics
from net_train.train.checkpoint import CheckpointManager
from net_train.utils.io import append_csv_row



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
    )

    total = w_extent * extent_l + w_boundary * boundary_l

    info = {
        "loss": float(total.detach().item()),
        **extent_info,
        **boundary_info,
    }
    return total, info



def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    plan: RuntimePlan,
    train_cfg: Dict[str, Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    logger,
) -> Dict[str, float]:
    model.train()

    train_section = train_cfg.get("train", {}) or {}
    batch_section = train_section.get("batch", {}) or {}

    grad_clip = float(train_section.get("grad_clip_norm", 0.0))
    grad_accum = int(batch_section.get("grad_accum_steps", 1))
    log_every = int(train_section.get("log_every_n_steps", 20))
    log_smooth_window = int(train_section.get("log_smoothing_window", log_every if log_every > 0 else 20))
    log_smooth_window = max(1, log_smooth_window)

    loss_cfg = train_cfg.get("loss", {}) or {}

    running_loss = 0.0
    steps = 0
    recent_loss = deque(maxlen=log_smooth_window)
    recent_extent = deque(maxlen=log_smooth_window)
    recent_boundary = deque(maxlen=log_smooth_window)

    optimizer.zero_grad(set_to_none=True)

    for step_idx, batch in enumerate(loader, start=1):
        batch = _to_device(batch, device=torch.device(plan.device))

        with _autocast_context(plan):
            out = model(batch["image"])
            loss, info = _loss_and_components(out, batch, loss_cfg)
            loss_scaled = loss / max(1, grad_accum)

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

        if step_idx % grad_accum == 0:
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

        steps += 1
        running_loss += float(info["loss"])
        recent_loss.append(float(info["loss"]))
        recent_extent.append(float(info["extent_total"]))
        recent_boundary.append(float(info["boundary_total"]))

        if log_every > 0 and step_idx % log_every == 0:
            avg_loss = sum(recent_loss) / max(1, len(recent_loss))
            avg_extent = sum(recent_extent) / max(1, len(recent_extent))
            avg_boundary = sum(recent_boundary) / max(1, len(recent_boundary))
            logger.info(
                f"epoch={epoch} step={step_idx}/{len(loader)} "
                f"loss={info['loss']:.5f} extent={info['extent_total']:.5f} boundary={info['boundary_total']:.5f} "
                f"| smooth{log_smooth_window}: loss={avg_loss:.5f} extent={avg_extent:.5f} boundary={avg_boundary:.5f}"
            )

    # flush leftover gradients
    if steps % grad_accum != 0:
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

    avg_loss = running_loss / max(1, steps)
    return {
        "train/loss": avg_loss,
    }


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    plan: RuntimePlan,
    train_cfg: Dict[str, Any],
) -> Dict[str, float]:
    model.eval()

    loss_cfg = train_cfg.get("loss", {}) or {}
    extent_cfg = ((train_cfg.get("metrics", {}) or {}).get("extent", {}) or {})
    boundary_cfg = ((train_cfg.get("metrics", {}) or {}).get("boundary", {}) or {})

    extent_threshold = float(extent_cfg.get("threshold", 0.5))
    boundary_thresholds = [float(v) for v in boundary_cfg.get("thresholds", [0.5, 0.35])]
    boundary_dilation = int(boundary_cfg.get("dilation_px", 2))

    dataset_cfg = train_cfg.get("dataset", {}) or {}
    targets_cfg = dataset_cfg.get("targets", {}) or {}
    extent_ignore = int((targets_cfg.get("extent", {}) or {}).get("ignore_value", 255))
    boundary_ignore = int((targets_cfg.get("boundary_bwbl", {}) or {}).get("ignore_value", 2))

    loss_sum = 0.0
    steps = 0

    # aggregate confusion terms for stable epoch-level metrics
    tp_e = fp_e = fn_e = 0.0

    boundary_f1_acc: Dict[float, List[float]] = {t: [] for t in boundary_thresholds}

    for batch in loader:
        batch = _to_device(batch, device=torch.device(plan.device))

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

        m_boundary = boundary_metrics_multi_threshold(
            logits=out["boundary_logits"],
            target=batch["boundary"],
            thresholds=boundary_thresholds,
            ignore_value=boundary_ignore,
            dilation_px=boundary_dilation,
        )
        for t in boundary_thresholds:
            boundary_f1_acc[t].append(float(m_boundary[f"boundary@{t:.2f}_f1"]))

    iou = float(tp_e / (tp_e + fp_e + fn_e)) if (tp_e + fp_e + fn_e) > 0 else 1.0
    precision = float(tp_e / (tp_e + fp_e)) if (tp_e + fp_e) > 0 else 1.0
    recall = float(tp_e / (tp_e + fn_e)) if (tp_e + fn_e) > 0 else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    out_metrics: Dict[str, float] = {
        "val/loss": float(loss_sum / max(1, steps)),
        "val/extent_iou": iou,
        "val/extent_f1": f1,
    }

    for t in boundary_thresholds:
        vals = boundary_f1_acc[t]
        out_metrics[f"val/boundary_f1@{t:.2f}"] = float(sum(vals) / max(1, len(vals)))

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
) -> List[Dict[str, float]]:
    train_section = train_cfg.get("train", {}) or {}
    epochs = int(train_section.get("epochs", 1))

    use_scaler = plan.device == "cuda" and plan.amp_enabled and plan.amp_dtype == "float16"
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            plan=plan,
            train_cfg=train_cfg,
            scaler=scaler if use_scaler else None,
            epoch=epoch,
            logger=logger,
        )

        val_metrics = validate_one_epoch(
            model=model,
            loader=val_loader,
            plan=plan,
            train_cfg=train_cfg,
        )

        if scheduler is not None:
            scheduler.step()

        row: Dict[str, float] = {"epoch": float(epoch)}
        row.update(train_metrics)
        row.update(val_metrics)
        row["lr"] = float(optimizer.param_groups[0]["lr"])

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

        logger.info(
            f"epoch={epoch}/{epochs} "
            f"train_loss={row['train/loss']:.5f} "
            f"val_loss={row['val/loss']:.5f} "
            f"val_extent_iou={row['val/extent_iou']:.4f} "
            f"best={ckpt_info['best_value']}"
        )

    return history
