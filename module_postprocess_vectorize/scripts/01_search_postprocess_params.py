from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from postprocess.pipeline import load_config
from postprocess.search import run_grid_search


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("postprocess_search")


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid search for post-processing parameters on validation predictions")
    ap.add_argument("--pred_root", required=True, help="Root directory with predicted rasters")
    ap.add_argument("--gt_root", required=True, help="GT polygons directory/file OR GT label raster directory/file")
    ap.add_argument("--config", default=str(ROOT / "configs" / "postprocess_config.yaml"))
    ap.add_argument("--params_override", default=None, help="Optional YAML with config overrides")
    ap.add_argument("--output_dir", required=True, help="Where to save best_params.yaml + search_results.json")
    ap.add_argument("--gt_mode", default="auto", choices=["auto", "vector", "raster"])
    ap.add_argument("--max_trials", type=int, default=None)
    ap.add_argument("--manifest_name", default=None, help="Manifest name in pred dir, default from config")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logger = setup_logger(args.log_level)
    console = Console()

    cfg = load_config(
        config_path=Path(args.config),
        override_path=Path(args.params_override) if args.params_override else None,
    )
    if args.manifest_name:
        cfg["predict_manifest_name"] = str(args.manifest_name)

    out = run_grid_search(
        pred_root=Path(args.pred_root),
        gt_root=Path(args.gt_root),
        output_dir=Path(args.output_dir),
        base_config=cfg,
        gt_mode=args.gt_mode,
        max_trials=args.max_trials,
        logger=logger,
    )

    logger.info("Best trial: %s", out["best_trial"])
    logger.info("Best metrics: %s", out["best_metrics"])
    console.print(f"[bold green]DONE[/bold green] {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
