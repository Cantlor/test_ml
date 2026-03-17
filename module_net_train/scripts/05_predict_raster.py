from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(REPO_ROOT))

from net_train.config import load_hardware_config, load_train_config, resolve_run_train_config_path
from net_train.data import load_stats_npz
from net_train.hardware import apply_torch_runtime_flags, build_runtime_plan
from net_train.infer import predict_aoi_raster
from net_train.models import build_model
from net_train.train import load_checkpoint
from net_train.utils.io import write_json
from net_train.utils.logging import setup_logger
from net_train.utils.progress import progress_enabled


def _resolve_cli_config_path(config_arg: str | None) -> Path:
    if config_arg is not None:
        return Path(config_arg).resolve()
    return (ROOT / "configs" / "train_config.yaml").resolve()


def _run_postprocess(
    *,
    extent_prob_path: Path,
    boundary_prob_path: Path,
    raster_path: Path,
    output_dir: Path,
    postprocess_config: Path,
    postprocess_params_override: Path | None,
    nodata_value: float | None,
    nodata_rule: str,
    control_band_1based: int,
    sample_id: str,
    predict_manifest_path: Path,
    logger,
    show_progress: bool | None,
) -> dict[str, str]:
    from module_postprocess_vectorize.postprocess.pipeline import (
        load_config as load_postprocess_config,
        run_postprocess_pipeline,
    )

    cfg = load_postprocess_config(
        config_path=postprocess_config.resolve(),
        override_path=postprocess_params_override.resolve() if postprocess_params_override is not None else None,
    )
    outputs = run_postprocess_pipeline(
        extent_prob_path=extent_prob_path,
        boundary_prob_path=boundary_prob_path,
        output_dir=output_dir.resolve(),
        config=cfg,
        valid_mask_path=None,
        footprint_path=raster_path.resolve(),
        footprint_nodata_value=nodata_value,
        footprint_nodata_rule=nodata_rule,
        footprint_control_band_1based=control_band_1based,
        input_context={
            "mode": "direct_raster",
            "sample_id": sample_id,
            "predict_manifest_path": str(predict_manifest_path.resolve()),
            "footprint_path": str(raster_path.resolve()),
        },
        gt_path=None,
        save_outputs=True,
        logger=logger,
        show_progress=show_progress,
    )
    return {
        k: v
        for k, v in outputs.items()
        if isinstance(v, str) and k.endswith(("_tif", "_gpkg", "_shp", "_json"))
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Predict extent/boundary probabilities for one raster using an existing module_net_train run",
    )
    ap.add_argument("--raster", required=True, help="Path to input GeoTIFF")
    ap.add_argument("--run_dir", required=True, help="Path to trained run directory with checkpoints and band_stats.npz")
    ap.add_argument("--output_dir", required=True, help="Directory where prediction outputs are written")
    ap.add_argument("--config", default=None, help="Optional train config path; run_dir/config_resolved.yaml still has priority")
    ap.add_argument("--hardware", default=str(ROOT / "configs/hardware_config.yaml"))
    ap.add_argument("--checkpoint", default=None, help="Optional checkpoint path (.pt); default is run_dir/checkpoints/best.pt")
    ap.add_argument("--sample_id", default=None, help="Logical sample name for manifest metadata")
    ap.add_argument("--dataset_key", default=None, help="Alias for --sample_id")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    ap.add_argument("--window_size", type=int, default=None, help="Override inference.tiling.window_size")
    ap.add_argument("--stride", type=int, default=None, help="Override inference.tiling.stride")
    ap.add_argument("--blend", default=None, help="Override inference.tiling.blend (mean|gaussian)")
    ap.add_argument("--gaussian_sigma", type=float, default=None, help="Override inference.tiling.gaussian_sigma")
    ap.add_argument("--gaussian_min_weight", type=float, default=None, help="Override inference.tiling.gaussian_min_weight")
    ap.add_argument(
        "--invalid_edge_guard_px",
        type=int,
        default=None,
        help="Override inference.tiling.invalid_edge_guard.radius_px",
    )
    ap.add_argument(
        "--invalid_edge_extent_scale",
        type=float,
        default=None,
        help="Override inference.tiling.invalid_edge_guard.extent_scale (0..1)",
    )
    ap.add_argument(
        "--invalid_edge_boundary_scale",
        type=float,
        default=None,
        help="Override inference.tiling.invalid_edge_guard.boundary_scale (0..1)",
    )
    ap.add_argument("--with-postprocess", action="store_true", help="Run postprocess pipeline after prediction")
    ap.add_argument(
        "--postprocess-config",
        default=str(REPO_ROOT / "module_postprocess_vectorize/configs/postprocess_config.yaml"),
    )
    ap.add_argument("--postprocess-params-override", default=None)
    ap.add_argument("--postprocess-output-dir", default=None, help="Default: <output_dir>/postprocess")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    console = Console()
    logger = setup_logger("predict_raster", level=args.log_level)
    show_progress = progress_enabled(True)

    run_dir = Path(args.run_dir).resolve()
    raster_path = Path(args.raster).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")

    if args.sample_id and args.dataset_key and args.sample_id != args.dataset_key:
        raise ValueError("--sample_id and --dataset_key are both set and differ; provide only one logical name")
    sample_id = args.sample_id or args.dataset_key or raster_path.stem

    train_config_path, used_run_config = resolve_run_train_config_path(
        _resolve_cli_config_path(args.config),
        run_dir,
    )
    train_cfg = load_train_config(train_config_path)
    hw_cfg = load_hardware_config(args.hardware)
    if used_run_config:
        logger.info(f"Using run-resolved config: {train_config_path}")
    elif args.config is None:
        logger.warning(
            f"config_resolved.yaml not found in run_dir; using default config: {train_config_path}"
        )
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
    tiling_cfg = infer_cfg.get("tiling", {}) or {}
    out_cfg = infer_cfg.get("outputs", {}) or {}
    ds_cfg = train_cfg.raw.get("dataset", {}) or {}
    inputs_cfg = ds_cfg.get("inputs", {}) or {}
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

    extent_name = str(out_cfg.get("extent_prob_name", "extent_prob.tif"))
    boundary_name = str(out_cfg.get("boundary_prob_name", "boundary_prob.tif"))
    extent_path = output_dir / extent_name
    boundary_path = output_dir / boundary_name
    manifest_path = output_dir / "predict_manifest.json"

    if not args.overwrite:
        collisions = [p for p in (extent_path, boundary_path, manifest_path) if p.exists()]
        if collisions:
            msg = ", ".join(str(p) for p in collisions)
            raise RuntimeError(
                f"Output files already exist: {msg}. Use --overwrite to replace existing artifacts."
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    infer_batch_cfg = tiling_cfg.get("batch_size", "auto")
    infer_batch = int(plan.batch_size if infer_batch_cfg == "auto" else infer_batch_cfg)
    tiling_guard_cfg = tiling_cfg.get("invalid_edge_guard", {}) or {}
    guard_enabled = bool(tiling_guard_cfg.get("enabled", False))

    window_size = int(args.window_size if args.window_size is not None else tiling_cfg.get("window_size", 512))
    stride = int(args.stride if args.stride is not None else tiling_cfg.get("stride", 384))
    blend = str(args.blend if args.blend is not None else tiling_cfg.get("blend", "mean"))
    gaussian_sigma = float(
        args.gaussian_sigma if args.gaussian_sigma is not None else tiling_cfg.get("gaussian_sigma", 0.30)
    )
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
        aoi_raster_path=raster_path,
        norm_stats=norm_stats,
        out_extent_path=extent_path,
        out_boundary_path=boundary_path,
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
        show_progress=show_progress,
    )

    pred_info["sample_id"] = sample_id
    pred_info["dataset_key"] = sample_id
    pred_info["input_mode"] = "direct_raster"
    pred_info["source_raster"] = str(raster_path)
    pred_info["input_raster"] = str(raster_path)
    pred_info["run_dir"] = str(run_dir)
    pred_info["checkpoint"] = str(ckpt_path)
    pred_info["checkpoint_fallback_used"] = bool(checkpoint_fallback_used)
    pred_info["config_used"] = str(train_config_path)

    if args.with_postprocess:
        pp_output_dir = (
            Path(args.postprocess_output_dir).resolve()
            if args.postprocess_output_dir is not None
            else (output_dir / "postprocess").resolve()
        )
        postprocess_outputs = _run_postprocess(
            extent_prob_path=extent_path,
            boundary_prob_path=boundary_path,
            raster_path=raster_path,
            output_dir=pp_output_dir,
            postprocess_config=Path(args.postprocess_config),
            postprocess_params_override=(
                Path(args.postprocess_params_override)
                if args.postprocess_params_override is not None
                else None
            ),
            nodata_value=nodata_value,
            nodata_rule=nodata_rule,
            control_band_1based=control_band_1based,
            sample_id=sample_id,
            predict_manifest_path=manifest_path,
            logger=logger,
            show_progress=show_progress,
        )
        pred_info["postprocess"] = postprocess_outputs

    write_json(manifest_path, pred_info)
    logger.info(f"Prediction written to: {output_dir}")
    console.print(f"[bold green]DONE[/bold green] {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
