from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from postprocess.pipeline import load_config, run_postprocess_pipeline


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
    ap.add_argument("--extent_prob", required=True)
    ap.add_argument("--boundary_prob", required=True)
    ap.add_argument("--valid_mask", default=None)
    ap.add_argument("--footprint", default=None)
    ap.add_argument("--gt", default=None, help="Optional GT polygons/raster for metrics_postproc.json")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--config", default=str(ROOT / "configs" / "postprocess_config.yaml"))
    ap.add_argument("--params_override", default=None, help="Optional YAML with tuned params (e.g. best_params.yaml)")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logger = setup_logger(args.log_level)
    console = Console()

    cfg = load_config(
        config_path=Path(args.config),
        override_path=Path(args.params_override) if args.params_override else None,
    )

    out = run_postprocess_pipeline(
        extent_prob_path=Path(args.extent_prob),
        boundary_prob_path=Path(args.boundary_prob),
        valid_mask_path=Path(args.valid_mask) if args.valid_mask else None,
        footprint_path=Path(args.footprint) if args.footprint else None,
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
