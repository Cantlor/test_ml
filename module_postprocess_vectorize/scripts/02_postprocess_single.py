from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

MODULE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MODULE_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from module_postprocess_vectorize.postprocess.inputs import resolve_prediction_sample_from_manifest, resolve_prediction_sample_from_run
from module_postprocess_vectorize.postprocess.pipeline import load_config, run_postprocess_pipeline


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("postprocess_single")


def main() -> int:
    ap = argparse.ArgumentParser(description="Postprocess one extent/boundary prediction pair into field polygons")
    ap.add_argument("--extent_prob", default=None)
    ap.add_argument("--boundary_prob", default=None)
    ap.add_argument("--valid_mask", default=None)
    ap.add_argument("--footprint", default=None)
    ap.add_argument("--predict_manifest", default=None, help="Current module_net_train predict_manifest.json")
    ap.add_argument("--run_dir", default=None, help="Run directory under output_data/module_net_train/runs")
    ap.add_argument("--dataset_key", default=None, help="Required when --run_dir contains multiple prediction samples")
    ap.add_argument("--gt", default=None, help="Optional GT polygons/raster for metrics_postproc.json")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--config", default=str(MODULE_ROOT / "configs" / "postprocess_config.yaml"))
    ap.add_argument("--params_override", default=None, help="Optional YAML with tuned params (e.g. best_params.yaml)")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logger = setup_logger(args.log_level)
    console = Console()

    cfg = load_config(
        config_path=Path(args.config),
        override_path=Path(args.params_override) if args.params_override else None,
    )

    explicit_pair = bool(args.extent_prob or args.boundary_prob)
    if explicit_pair and not (args.extent_prob and args.boundary_prob):
        raise ValueError("--extent_prob and --boundary_prob must be passed together")

    resolver_modes = sum(
        1
        for flag in (explicit_pair, bool(args.predict_manifest), bool(args.run_dir))
        if flag
    )
    if resolver_modes != 1:
        raise ValueError(
            "Choose exactly one input mode: explicit --extent_prob/--boundary_prob, --predict_manifest, or --run_dir"
        )

    sample = None
    if args.predict_manifest:
        sample = resolve_prediction_sample_from_manifest(
            Path(args.predict_manifest),
            extent_name=str(cfg.get("extent_prob_name", "extent_prob.tif")),
            boundary_name=str(cfg.get("boundary_prob_name", "boundary_prob.tif")),
            valid_name=str(cfg.get("valid_mask_name", "valid_mask.tif")),
        )
    elif args.run_dir:
        sample = resolve_prediction_sample_from_run(
            Path(args.run_dir),
            dataset_key=args.dataset_key,
            extent_name=str(cfg.get("extent_prob_name", "extent_prob.tif")),
            boundary_name=str(cfg.get("boundary_prob_name", "boundary_prob.tif")),
            valid_name=str(cfg.get("valid_mask_name", "valid_mask.tif")),
            manifest_name=str(cfg.get("predict_manifest_name", "predict_manifest.json")),
        )

    extent_prob_path = Path(args.extent_prob) if args.extent_prob else sample.extent_prob_path
    boundary_prob_path = Path(args.boundary_prob) if args.boundary_prob else sample.boundary_prob_path
    valid_mask_path = Path(args.valid_mask) if args.valid_mask else (sample.valid_mask_path if sample is not None else None)
    footprint_path = Path(args.footprint) if args.footprint else (sample.footprint_path if sample is not None else None)
    footprint_nodata_value = sample.valid_nodata_value if sample is not None else None
    footprint_nodata_rule = sample.valid_nodata_rule if sample is not None else "control-band"
    footprint_control_band_1based = sample.valid_control_band_1based if sample is not None else 1
    input_context = sample.to_input_context() if sample is not None else {
        "mode": "explicit_paths",
        "extent_prob_path": str(extent_prob_path.resolve()),
        "boundary_prob_path": str(boundary_prob_path.resolve()),
        "valid_mask_path": str(valid_mask_path.resolve()) if valid_mask_path is not None else None,
        "footprint_path": str(footprint_path.resolve()) if footprint_path is not None else None,
    }

    out = run_postprocess_pipeline(
        extent_prob_path=extent_prob_path,
        boundary_prob_path=boundary_prob_path,
        valid_mask_path=valid_mask_path,
        footprint_path=footprint_path,
        footprint_nodata_value=footprint_nodata_value,
        footprint_nodata_rule=footprint_nodata_rule,
        footprint_control_band_1based=footprint_control_band_1based,
        input_context=input_context,
        gt_path=Path(args.gt) if args.gt else None,
        output_dir=Path(args.output_dir),
        config=cfg,
        save_outputs=True,
        logger=logger,
    )

    logger.info("Labels stats: %s", out["labels_stats"])
    if out.get("metrics_postproc") is not None:
        logger.info("Postprocess metrics: %s", out["metrics_postproc"])

    console.print(f"[bold green]DONE[/bold green] {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
