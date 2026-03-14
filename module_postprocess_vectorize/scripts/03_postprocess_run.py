from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from postprocess.io import ensure_dir, write_json
from postprocess.pipeline import load_config, run_postprocess_pipeline
from postprocess.search import discover_prediction_samples


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("postprocess_run")


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply tuned postprocessing to all predictions in run_dir")
    ap.add_argument("--run_dir", required=True, help="Path to specific run directory inside output_data/module_net_train/runs")
    ap.add_argument("--pred_root", default=None, help="Default: <run_dir>/pred")
    ap.add_argument("--output_root", default=None, help="Default: <run_dir>/postprocess")

    ap.add_argument("--config", default=str(ROOT / "configs" / "postprocess_config.yaml"))
    ap.add_argument("--params_override", default=None, help="Optional YAML with tuned params")

    ap.add_argument("--extent_name", default=None)
    ap.add_argument("--boundary_name", default=None)
    ap.add_argument("--valid_name", default=None)
    ap.add_argument("--manifest_name", default=None, help="Manifest in pred dir with AOI/source raster path")

    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logger = setup_logger(args.log_level)
    console = Console()

    run_dir = Path(args.run_dir).resolve()
    pred_root = Path(args.pred_root).resolve() if args.pred_root else (run_dir / "pred")
    output_root = Path(args.output_root).resolve() if args.output_root else (run_dir / "postprocess")
    ensure_dir(output_root)

    cfg = load_config(
        config_path=Path(args.config),
        override_path=Path(args.params_override) if args.params_override else None,
    )

    extent_name = args.extent_name or str(cfg.get("extent_prob_name", "extent_prob.tif"))
    boundary_name = args.boundary_name or str(cfg.get("boundary_prob_name", "boundary_prob.tif"))
    valid_name = args.valid_name or str(cfg.get("valid_mask_name", "valid_mask.tif"))
    manifest_name = args.manifest_name or str(cfg.get("predict_manifest_name", "predict_manifest.json"))

    samples = discover_prediction_samples(
        pred_root=pred_root,
        extent_name=extent_name,
        boundary_name=boundary_name,
        valid_name=valid_name,
        manifest_name=manifest_name,
    )
    if not samples:
        raise RuntimeError(f"No prediction samples found under {pred_root}")

    summary = {
        "run_dir": str(run_dir),
        "pred_root": str(pred_root),
        "output_root": str(output_root),
        "num_samples": len(samples),
        "samples": [],
    }

    for sample in tqdm(samples, desc="postprocess-run", unit="sample"):
        out_dir = output_root / sample.sample_id
        try:
            res = run_postprocess_pipeline(
                extent_prob_path=sample.extent_prob_path,
                boundary_prob_path=sample.boundary_prob_path,
                valid_mask_path=sample.valid_mask_path,
                footprint_path=sample.footprint_path,
                output_dir=out_dir,
                config=cfg,
                save_outputs=True,
                logger=logger,
            )
            summary["samples"].append(
                {
                    "sample_id": sample.sample_id,
                    "status": "ok",
                    "extent_prob": str(sample.extent_prob_path),
                    "boundary_prob": str(sample.boundary_prob_path),
                    "valid_mask": str(sample.valid_mask_path) if sample.valid_mask_path else None,
                    "footprint": str(sample.footprint_path) if sample.footprint_path else None,
                    "output_dir": str(out_dir),
                    "labels_stats": res.get("labels_stats"),
                }
            )
        except Exception as exc:
            summary["samples"].append(
                {
                    "sample_id": sample.sample_id,
                    "status": "error",
                    "error": str(exc),
                }
            )
            if not args.continue_on_error:
                raise

    write_json(output_root / "postprocess_run_summary.json", summary)
    console.print(f"[bold green]DONE[/bold green] {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
