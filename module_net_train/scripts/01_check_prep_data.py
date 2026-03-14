from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from net_train.config import load_train_config
from net_train.data.index import build_index
from net_train.utils.io import write_json
from net_train.utils.logging import setup_logger


def _meta_stats(values: List[float]) -> Dict[str, float | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def _check_mask_values(mask_path: Path, allowed: set[int]) -> Dict[str, object]:
    with rasterio.open(mask_path) as ds:
        arr = ds.read(1)
    uniq = set(np.unique(arr).tolist())
    bad = sorted(v for v in uniq if int(v) not in allowed)
    return {
        "path": str(mask_path),
        "unique_values": sorted(int(v) for v in uniq),
        "bad_values": bad,
        "ok": len(bad) == 0,
    }


def _check_img_shape(img_path: Path, expected_bands: int) -> Dict[str, object]:
    with rasterio.open(img_path) as ds:
        count = int(ds.count)
        dtype = ds.dtypes[0] if ds.dtypes else "unknown"
        nodata = ds.nodata
    return {
        "path": str(img_path),
        "bands": count,
        "dtype": str(dtype),
        "nodata": None if nodata is None else float(nodata),
        "ok": count >= expected_bands,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/train_config.yaml"))
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--max_mask_checks", type=int, default=24)
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()

    logger = setup_logger("check_prep_data", level=args.log_level)
    console = Console()

    cfg = load_train_config(args.config)
    prep_data_root = cfg.paths.get("prep_data_root")
    if prep_data_root is None:
        raise RuntimeError("paths.prep_data_root is missing in config")

    splits_cfg = ((cfg.raw.get("dataset", {}) or {}).get("splits", {}) or {})
    splits = {
        "train": str(splits_cfg.get("train", "train")),
        "val": str(splits_cfg.get("val", "validation")),
        "test": str(splits_cfg.get("test", "test")),
    }

    dataset_cfg = cfg.raw.get("dataset", {}) or {}
    inputs_cfg = dataset_cfg.get("inputs", {}) or {}
    targets_cfg = dataset_cfg.get("targets", {}) or {}
    extent_cfg = targets_cfg.get("extent", {}) or {}
    bwbl_cfg = targets_cfg.get("boundary_bwbl", {}) or {}

    expected_bands = int(inputs_cfg.get("num_bands", 8))
    extent_ignore = int(extent_cfg.get("ignore_value", 255))
    bwbl_ignore = int(bwbl_cfg.get("ignore_value", 2))

    logger.info(f"Config: {cfg.config_path}")
    logger.info(f"prep_data_root: {prep_data_root}")

    index = build_index(prep_data_root=prep_data_root, splits=splits)

    report: Dict[str, object] = {
        "config": str(cfg.config_path),
        "prep_data_root": str(prep_data_root),
        "splits": {},
        "errors": [],
        "warnings": [],
        "ok": True,
    }

    total_samples = 0

    for alias, res in index.items():
        records = res.records
        total_samples += len(records)

        split_errors: List[str] = []
        split_warnings: List[str] = []

        if res.missing_files:
            split_errors.extend(res.missing_files)

        valid_vals = [float(r.meta.get("valid_ratio")) for r in records if r.meta.get("valid_ratio") is not None]
        mask_vals = [float(r.meta.get("mask_ratio")) for r in records if r.meta.get("mask_ratio") is not None]
        edge_vals = [float(r.meta.get("edge_ratio")) for r in records if r.meta.get("edge_ratio") is not None]

        datasets = {}
        for r in records:
            datasets[r.dataset] = datasets.get(r.dataset, 0) + 1

        image_checks = []
        for r in records[: min(8, len(records))]:
            ch = _check_img_shape(r.img_path, expected_bands)
            image_checks.append(ch)
            if not ch["ok"]:
                split_errors.append(
                    f"{r.img_path}: bands={ch['bands']} expected>={expected_bands}"
                )

        extent_allowed = {0, 1, extent_ignore}
        bwbl_allowed = {0, 1, bwbl_ignore}
        valid_allowed = {0, 1}
        mask_checks = []
        mask_candidates = records[: min(args.max_mask_checks, len(records))]
        for r in mask_candidates:
            e_chk = _check_mask_values(r.extent_path, extent_allowed)
            b_chk = _check_mask_values(r.boundary_bwbl_path, bwbl_allowed)
            v_chk = _check_mask_values(r.valid_path, valid_allowed)
            mask_checks.append({"extent": e_chk, "boundary_bwbl": b_chk, "valid": v_chk})
            if not e_chk["ok"]:
                split_errors.append(
                    f"{r.extent_path}: bad extent values {e_chk['bad_values']}"
                )
            if not b_chk["ok"]:
                split_errors.append(
                    f"{r.boundary_bwbl_path}: bad bwbl values {b_chk['bad_values']}"
                )
            if not v_chk["ok"]:
                split_errors.append(
                    f"{r.valid_path}: bad valid values {v_chk['bad_values']}"
                )

        split_report = {
            "split_dir": splits[alias],
            "samples": len(records),
            "datasets": datasets,
            "meta_stats": {
                "valid_ratio": _meta_stats(valid_vals),
                "mask_ratio": _meta_stats(mask_vals),
                "edge_ratio": _meta_stats(edge_vals),
            },
            "checked_files": {
                "img_samples": len(image_checks),
                "mask_samples": len(mask_checks),
            },
            "errors": split_errors,
            "warnings": split_warnings,
        }

        report["splits"][alias] = split_report

        if split_errors:
            report["errors"].extend([f"[{alias}] {e}" for e in split_errors])

    report["ok"] = len(report["errors"]) == 0 and total_samples > 0

    console.print("[bold]01_check_prep_data[/bold]")
    for alias in ["train", "val", "test"]:
        sr = report["splits"].get(alias, {})
        console.print(
            f"{alias}: samples={sr.get('samples', 0)} "
            f"errors={len(sr.get('errors', []))}"
        )
    console.print(f"total_samples={total_samples}")

    if args.out_json:
        out_json = Path(args.out_json)
    else:
        runs_root = cfg.paths.get("runs_root", (cfg.project_root / "output_data/module_net_train/runs").resolve())
        out_json = runs_root / "prep_data_summary.json"

    write_json(out_json, report)
    console.print(f"report: {out_json}")

    if report["ok"]:
        console.print("[bold green]OK[/bold green]")
        return 0

    console.print(f"[bold red]FAILED[/bold red] errors={len(report['errors'])}")
    logger.error("First errors:")
    for e in report["errors"][:10]:
        logger.error(f"- {e}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
