from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from postprocess.io import write_json
from postprocess.metrics import evaluate_polygons, load_polygons


def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger("postprocess_eval")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate polygons (object-level F1 @ IoU threshold)")
    ap.add_argument("--gt", required=True, help="GT vector or GT label/mask raster")
    ap.add_argument("--pred", required=True, help="Prediction vector or label raster")
    ap.add_argument("--iou_threshold", type=float, default=0.5)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logger = setup_logger(args.log_level)
    console = Console()

    gt_gdf = load_polygons(Path(args.gt))
    pred_gdf = load_polygons(Path(args.pred))

    metrics = evaluate_polygons(
        gt_gdf=gt_gdf,
        pred_gdf=pred_gdf,
        iou_threshold=float(args.iou_threshold),
    )

    logger.info("Metrics: %s", metrics)

    if args.out_json:
        out = Path(args.out_json).resolve()
        write_json(out, metrics)
        console.print(f"[bold green]DONE[/bold green] {out}")
    else:
        console.print(metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
