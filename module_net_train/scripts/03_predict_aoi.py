from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from net_train.config import load_hardware_config, load_train_config, resolve_inference_manifest_path
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
    ap.add_argument("--run_dir", required=True, help="Path to runs/<run_id>")
    ap.add_argument("--checkpoint", default=None, help="Path to checkpoint (.pt). Default: run_dir/checkpoints/best.pt")
    ap.add_argument("--dataset_key", default=None)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    console = Console()
    logger = setup_logger("predict_aoi", level=args.log_level)

    train_cfg = load_train_config(args.config)
    hw_cfg = load_hardware_config(args.hardware)

    run_dir = Path(args.run_dir).resolve()
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
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / "last.pt"
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

    pred_info = predict_aoi_raster(
        model=model,
        plan=plan,
        aoi_raster_path=aoi_path,
        norm_stats=norm_stats,
        out_extent_path=pred_dir / str(out_cfg.get("extent_prob_name", "extent_prob.tif")),
        out_boundary_path=pred_dir / str(out_cfg.get("boundary_prob_name", "boundary_prob.tif")),
        window_size=int(tiling_cfg.get("window_size", 512)),
        stride=int(tiling_cfg.get("stride", 384)),
        batch_size=max(1, infer_batch),
        blend=str(tiling_cfg.get("blend", "mean")),
        out_dtype=str(out_cfg.get("dtype", "float32")),
        compress=str(out_cfg.get("compress", "DEFLATE")).upper(),
        tiled=bool(out_cfg.get("tiled", True)),
        bigtiff=str(out_cfg.get("bigtiff", "if_needed")),
        num_bands=num_bands,
        add_valid_channel=add_valid_channel,
        nodata_value=nodata_value,
        nodata_rule=nodata_rule,
        control_band_1based=control_band_1based,
    )

    pred_info["checkpoint"] = str(ckpt_path)
    write_json(pred_dir / "predict_manifest.json", pred_info)

    logger.info(f"Prediction written to: {pred_dir}")
    console.print(f"[bold green]DONE[/bold green] {pred_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
