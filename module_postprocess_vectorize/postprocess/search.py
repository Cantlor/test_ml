from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import geopandas as gpd
from tqdm import tqdm

from .io import deep_update, read_json, to_serializable, write_json, write_yaml
from .metrics import aggregate_metrics, evaluate_polygons, load_polygons, ranking_key
from .pipeline import run_postprocess_pipeline


@dataclass(frozen=True)
class PredictionSample:
    sample_id: str
    extent_prob_path: Path
    boundary_prob_path: Path
    valid_mask_path: Optional[Path]
    footprint_path: Optional[Path]


def _resolve_footprint_from_manifest(pred_dir: Path, manifest_name: str = "predict_manifest.json") -> Optional[Path]:
    manifest_path = pred_dir / manifest_name
    if not manifest_path.exists():
        return None
    try:
        obj = read_json(manifest_path)
    except Exception:
        return None

    for key in ("aoi_raster", "source_raster", "input_raster"):
        p = obj.get(key)
        if isinstance(p, str):
            cand = Path(p).resolve()
            if cand.exists():
                return cand
    return None


def discover_prediction_samples(
    pred_root: Path,
    extent_name: str = "extent_prob.tif",
    boundary_name: str = "boundary_prob.tif",
    valid_name: str = "valid_mask.tif",
    manifest_name: str = "predict_manifest.json",
) -> List[PredictionSample]:
    root = pred_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Prediction root does not exist: {root}")

    extent_paths: List[Path] = []
    direct_extent = root / extent_name
    if direct_extent.exists():
        extent_paths = [direct_extent]
    else:
        extent_paths = sorted(root.rglob(extent_name))

    samples: List[PredictionSample] = []
    for extent_path in extent_paths:
        pred_dir = extent_path.parent
        boundary_path = pred_dir / boundary_name
        if not boundary_path.exists():
            continue

        valid_path = pred_dir / valid_name
        valid_mask = valid_path if valid_path.exists() else None
        footprint_path = None if valid_mask is not None else _resolve_footprint_from_manifest(pred_dir, manifest_name=manifest_name)

        if pred_dir == root:
            sample_id = pred_dir.name
        else:
            sample_id = str(pred_dir.relative_to(root))

        samples.append(
            PredictionSample(
                sample_id=sample_id,
                extent_prob_path=extent_path.resolve(),
                boundary_prob_path=boundary_path.resolve(),
                valid_mask_path=(valid_mask.resolve() if valid_mask is not None else None),
                footprint_path=footprint_path,
            )
        )

    samples = sorted(samples, key=lambda s: s.sample_id)
    return samples


def _candidate_gt_paths(gt_root: Path, sample_id: str, mode: str) -> List[Path]:
    sid = Path(sample_id)
    leaf = sid.name

    vec_exts = [".gpkg", ".shp", ".geojson", ".json"]
    ras_exts = [".tif", ".tiff"]

    if mode == "vector":
        exts = vec_exts
    elif mode == "raster":
        exts = ras_exts
    else:
        exts = vec_exts + ras_exts

    candidates: List[Path] = []
    for ext in exts:
        candidates.extend(
            [
                gt_root / f"{sample_id}{ext}",
                gt_root / sample_id / f"{leaf}{ext}",
                gt_root / sample_id / f"gt{ext}",
                gt_root / f"{leaf}{ext}",
                gt_root / leaf / f"{leaf}{ext}",
                gt_root / leaf / f"gt{ext}",
            ]
        )

    uniq: List[Path] = []
    seen = set()
    for c in candidates:
        rc = c.resolve()
        if rc in seen:
            continue
        seen.add(rc)
        uniq.append(rc)
    return uniq


def resolve_gt_path(sample_id: str, gt_root: Path, gt_mode: str = "auto") -> Path:
    root = gt_root.resolve()
    if root.is_file():
        return root

    candidates = _candidate_gt_paths(root, sample_id, gt_mode)
    found = [p for p in candidates if p.exists() and p.is_file()]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        # deterministic choice by shortest path then lexical ordering.
        found = sorted(found, key=lambda p: (len(str(p)), str(p)))
        return found[0]

    # fallback: search by leaf name
    leaf = Path(sample_id).name
    wildcard = list(root.rglob(f"{leaf}.*"))
    allowed = {".gpkg", ".shp", ".geojson", ".json", ".tif", ".tiff"}
    wildcard = [p for p in wildcard if p.suffix.lower() in allowed and p.is_file()]
    if wildcard:
        wildcard = sorted(wildcard, key=lambda p: (len(str(p)), str(p)))
        return wildcard[0].resolve()

    raise FileNotFoundError(f"GT file not found for sample_id='{sample_id}' under {root}")


def build_grid(base_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    search_cfg = (base_config.get("search", {}) or {})
    grid_cfg = (search_cfg.get("grid", {}) or {})

    default_grid = {
        "extent_thr": [0.45, 0.5, 0.55],
        "boundary_thr": [0.35, 0.45, 0.55],
        "boundary_dilate_px": [0, 1, 2],
        "gaussian_sigma_px": [0.6, 1.0, 1.4],
        "min_area_m2": [80.0, 120.0, 180.0],
        "simplify_m": [0.4, 0.8, 1.2],
        "marker_erode_px": [0, 1, 2],
        "seed_min_distance_px": [4, 6, 8],
    }

    if not grid_cfg:
        grid_cfg = default_grid

    keys = list(grid_cfg.keys())
    values = []
    for k in keys:
        raw = grid_cfg[k]
        if isinstance(raw, list):
            values.append(raw)
        else:
            values.append([raw])

    combinations = []
    for product in itertools.product(*values):
        params = {k: v for k, v in zip(keys, product)}
        combinations.append(params)
    return combinations


def run_grid_search(
    *,
    pred_root: Path,
    gt_root: Path,
    output_dir: Path,
    base_config: Mapping[str, Any],
    gt_mode: str = "auto",
    max_trials: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    log = logger or logging.getLogger(__name__)

    samples = discover_prediction_samples(
        pred_root=pred_root,
        extent_name=str(base_config.get("extent_prob_name", "extent_prob.tif")),
        boundary_name=str(base_config.get("boundary_prob_name", "boundary_prob.tif")),
        valid_name=str(base_config.get("valid_mask_name", "valid_mask.tif")),
        manifest_name=str(base_config.get("predict_manifest_name", "predict_manifest.json")),
    )
    if not samples:
        raise RuntimeError(f"No prediction samples discovered under {pred_root}")

    gt_map: Dict[str, gpd.GeoDataFrame] = {}
    gt_path_map: Dict[str, str] = {}
    for s in samples:
        gt_path = resolve_gt_path(s.sample_id, gt_root=gt_root, gt_mode=gt_mode)
        gt_map[s.sample_id] = load_polygons(gt_path)
        gt_path_map[s.sample_id] = str(gt_path)

    grid = build_grid(base_config)
    if max_trials is not None:
        grid = grid[: max(1, int(max_trials))]

    if not grid:
        raise RuntimeError("Search grid is empty")

    iou_thr = float((base_config.get("scoring", {}) or {}).get("iou_threshold", 0.5))

    rows: List[Dict[str, Any]] = []
    for trial_idx, params in enumerate(tqdm(grid, desc="postprocess-grid", unit="trial"), start=1):
        trial_cfg = deep_update(dict(base_config), params)
        trial_cfg["save_intermediates"] = False
        trial_cfg["export_shp"] = False

        sample_metrics = []
        for s in samples:
            res = run_postprocess_pipeline(
                extent_prob_path=s.extent_prob_path,
                boundary_prob_path=s.boundary_prob_path,
                valid_mask_path=s.valid_mask_path,
                footprint_path=s.footprint_path,
                output_dir=output_dir / "_tmp" / f"trial_{trial_idx:04d}" / s.sample_id,
                config=trial_cfg,
                save_outputs=False,
                logger=log,
            )
            pred_gdf = res["fields_pred"]
            gt_gdf = gt_map[s.sample_id]
            m = evaluate_polygons(gt_gdf=gt_gdf, pred_gdf=pred_gdf, iou_threshold=iou_thr)
            m["sample_id"] = s.sample_id
            sample_metrics.append(m)

        aggregated = aggregate_metrics(sample_metrics)
        rows.append(
            {
                "trial": trial_idx,
                "params": params,
                "metrics": aggregated,
                "sample_metrics": sample_metrics,
            }
        )

    rows_sorted = sorted(rows, key=lambda r: ranking_key(r["metrics"]), reverse=True)
    best = rows_sorted[0]

    out_dir = output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    best_params = deep_update(dict(base_config), dict(best["params"]))
    write_yaml(out_dir / "best_params.yaml", best_params)

    summary = {
        "pred_root": str(pred_root.resolve()),
        "gt_root": str(gt_root.resolve()),
        "gt_mode": gt_mode,
        "num_samples": len(samples),
        "samples": [s.sample_id for s in samples],
        "gt_paths": gt_path_map,
        "num_trials": len(rows),
        "best_trial": best["trial"],
        "best_params_override": best["params"],
        "best_metrics": best["metrics"],
        "results": rows_sorted,
    }
    write_json(out_dir / "search_results.json", to_serializable(summary))

    return summary
