from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from net_train.config import (
    load_hardware_config,
    load_train_config,
    resolve_inference_manifest_path,
    resolve_run_train_config_path,
)
from net_train.data import load_stats_npz
from net_train.hardware import apply_torch_runtime_flags, build_runtime_plan
from net_train.infer import predict_aoi_raster, resolve_aoi_path
from net_train.models import build_model
from net_train.train import load_checkpoint
from net_train.utils.io import write_json
from net_train.utils.logging import setup_logger



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/train_config.yaml"))
    ap.add_argument("--hardware", default=str(ROOT / "configs/hardware_config.yaml"))
    ap.add_argument("--run_dir", required=True, help="Path to specific run directory inside output_data/module_net_train/runs")
    ap.add_argument("--checkpoint", default=None, help="Path to checkpoint (.pt). Default: run_dir/checkpoints/best.pt")
    ap.add_argument("--dataset_key", default=None)
    ap.add_argument("--window_size", type=int, default=None, help="Override inference.tiling.window_size")
    ap.add_argument("--stride", type=int, default=None, help="Override inference.tiling.stride")
    ap.add_argument("--blend", default=None, help="Override inference.tiling.blend (mean|gaussian)")
    ap.add_argument("--gaussian_sigma", type=float, default=None, help="Override inference.tiling.gaussian_sigma")
    ap.add_argument("--gaussian_min_weight", type=float, default=None, help="Override inference.tiling.gaussian_min_weight")
    ap.add_argument("--invalid_edge_guard_px", type=int, default=None, help="Override inference.tiling.invalid_edge_guard.radius_px")
    ap.add_argument("--invalid_edge_extent_scale", type=float, default=None, help="Override inference.tiling.invalid_edge_guard.extent_scale (0..1)")
    ap.add_argument("--invalid_edge_boundary_scale", type=float, default=None, help="Override inference.tiling.invalid_edge_guard.boundary_scale (0..1)")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    console = Console()
    logger = setup_logger("predict_aoi", level=args.log_level)

    run_dir = Path(args.run_dir).resolve()
    train_config_path, used_run_config = resolve_run_train_config_path(args.config, run_dir)
    train_cfg = load_train_config(train_config_path)
    hw_cfg = load_hardware_config(args.hardware)
    if used_run_config:
        logger.info(f"Using run-resolved config: {train_config_path}")
    else:
        logger.warning(
            f"config_resolved.yaml not found in run_dir; using CLI config: {train_config_path}"
        )

    stats_path = run_dir / "band_stats.npz"
    if not stats_path.exists():
        raise RuntimeError(f"Missing normalization stats: {stats_path}")
    norm_stats = load_stats_npz(stats_path)

    plan = build_runtime_plan(train_cfg, hw_cfg)
    apply_torch_runtime_flags(plan)

    infer_cfg = train_cfg.raw.get("inference", {}) or {}
    aoi_cfg = infer_cfg.get("aoi", {}) or {}
    tiling_cfg = infer_cfg.get("tiling", {}) or {}
    out_cfg = infer_cfg.get("outputs", {}) or {}
    ds_cfg = train_cfg.raw.get("dataset", {}) or {}
    inputs_cfg = (ds_cfg.get("inputs", {}) or {})
    num_bands = int(inputs_cfg.get("num_bands", 8))
    add_valid_channel = bool(inputs_cfg.get("add_valid_channel", True))
    nodata_value = inputs_cfg.get("nodata_value", 65536)
    nodata_value = None if nodata_value is None else float(nodata_value)
    nodata_rule = str(inputs_cfg.get("nodata_rule", "control-band"))
    control_band_1based = int(inputs_cfg.get("control_band_1based", 1))

    expected_in_channels = int(num_bands + (1 if add_valid_channel else 0))
    model_cfg = train_cfg.raw.setdefault("model", {})
    if int(model_cfg.get("in_channels", expected_in_channels)) != expected_in_channels:
        logger.warning(
            "model.in_channels mismatch, overriding to dataset contract value: %s",
            expected_in_channels,
        )
    model_cfg["in_channels"] = expected_in_channels

    model = build_model(train_cfg.raw.get("model", {}) or {})
    model = model.to(torch.device(plan.device))

    ckpt_path = Path(args.checkpoint).resolve() if args.checkpoint else (run_dir / "checkpoints" / "best.pt")
    checkpoint_fallback_used = False
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
        checkpoint_fallback_used = True
        logger.warning("best.pt not found, falling back to last.pt for prediction: %s", ckpt_path)
    if not ckpt_path.exists():
        raise RuntimeError(f"Checkpoint not found in {run_dir / 'checkpoints'}")

    ckpt = load_checkpoint(ckpt_path, map_location=plan.device)
    model.load_state_dict(ckpt["model_state"])

    dataset_key = args.dataset_key or str(aoi_cfg.get("dataset_key", "first_raster"))

    manifest_path = resolve_inference_manifest_path(train_cfg)
    aoi_path = resolve_aoi_path(manifest_path, dataset_key)

    pred_dir = run_dir / "pred" / dataset_key
    pred_dir.mkdir(parents=True, exist_ok=True)

    infer_batch_cfg = tiling_cfg.get("batch_size", "auto")
    infer_batch = int(plan.batch_size if infer_batch_cfg == "auto" else infer_batch_cfg)
    tiling_guard_cfg = tiling_cfg.get("invalid_edge_guard", {}) or {}
    guard_enabled = bool(tiling_guard_cfg.get("enabled", False))

    window_size = int(args.window_size if args.window_size is not None else tiling_cfg.get("window_size", 512))
    stride = int(args.stride if args.stride is not None else tiling_cfg.get("stride", 384))
    blend = str(args.blend if args.blend is not None else tiling_cfg.get("blend", "mean"))
    gaussian_sigma = float(args.gaussian_sigma if args.gaussian_sigma is not None else tiling_cfg.get("gaussian_sigma", 0.30))
    gaussian_min_weight = float(
        args.gaussian_min_weight
        if args.gaussian_min_weight is not None
        else tiling_cfg.get("gaussian_min_weight", 0.05)
    )
    guard_px_default = int(tiling_guard_cfg.get("radius_px", 0))
    guard_extent_default = float(tiling_guard_cfg.get("extent_scale", 1.0))
    guard_boundary_default = float(tiling_guard_cfg.get("boundary_scale", 1.0))

    guard_px = int(args.invalid_edge_guard_px if args.invalid_edge_guard_px is not None else guard_px_default)
    guard_extent_scale = float(
        args.invalid_edge_extent_scale if args.invalid_edge_extent_scale is not None else guard_extent_default
    )
    guard_boundary_scale = float(
        args.invalid_edge_boundary_scale if args.invalid_edge_boundary_scale is not None else guard_boundary_default
    )
    if not guard_enabled and args.invalid_edge_guard_px is None:
        guard_px = 0
        guard_extent_scale = 1.0
        guard_boundary_scale = 1.0

    logger.info(
        "Inference tiling: window=%s stride=%s blend=%s sigma=%.3f min_w=%.3f guard_px=%s extent_scale=%.2f boundary_scale=%.2f",
        window_size,
        stride,
        blend,
        gaussian_sigma,
        gaussian_min_weight,
        guard_px,
        guard_extent_scale,
        guard_boundary_scale,
    )

    pred_info = predict_aoi_raster(
        model=model,
        plan=plan,
        aoi_raster_path=aoi_path,
        norm_stats=norm_stats,
        out_extent_path=pred_dir / str(out_cfg.get("extent_prob_name", "extent_prob.tif")),
        out_boundary_path=pred_dir / str(out_cfg.get("boundary_prob_name", "boundary_prob.tif")),
        window_size=window_size,
        stride=stride,
        batch_size=max(1, infer_batch),
        blend=blend,
        gaussian_sigma=gaussian_sigma,
        gaussian_min_weight=gaussian_min_weight,
        out_dtype=str(out_cfg.get("dtype", "float32")),
        compress=str(out_cfg.get("compress", "DEFLATE")).upper(),
        tiled=bool(out_cfg.get("tiled", True)),
        bigtiff=str(out_cfg.get("bigtiff", "if_needed")),
        num_bands=num_bands,
        add_valid_channel=add_valid_channel,
        nodata_value=nodata_value,
        nodata_rule=nodata_rule,
        control_band_1based=control_band_1based,
        invalid_edge_guard_px=guard_px,
        invalid_edge_extent_scale=guard_extent_scale,
        invalid_edge_boundary_scale=guard_boundary_scale,
    )

    pred_info["checkpoint"] = str(ckpt_path)
    pred_info["checkpoint_fallback_used"] = bool(checkpoint_fallback_used)
    pred_info["config_used"] = str(train_config_path)
    write_json(pred_dir / "predict_manifest.json", pred_info)

    logger.info(f"Prediction written to: {pred_dir}")
    console.print(f"[bold green]DONE[/bold green] {pred_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
