from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd

from .io import deep_update, to_serializable, write_json, write_yaml
from .inputs import PredictionSample, discover_prediction_samples
from .metrics import aggregate_metrics, evaluate_polygons, load_polygons, ranking_key
from .pipeline import run_postprocess_pipeline
from .progress import iter_progress


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


def _grid_values(base_config: Mapping[str, Any]) -> Dict[str, List[Any]]:
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

    out: Dict[str, List[Any]] = {}
    for key, raw in grid_cfg.items():
        if isinstance(raw, list):
            out[str(key)] = list(raw)
        else:
            out[str(key)] = [raw]
    return out


def build_grid(base_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    values_map = _grid_values(base_config)
    keys = list(values_map.keys())
    values = [values_map[k] for k in keys]

    combinations = []
    for product in itertools.product(*values):
        params = {k: v for k, v in zip(keys, product)}
        combinations.append(params)
    return combinations


def _params_key(params: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((str(k), params[k]) for k in params.keys()))


def _sample_evenly(items: Sequence[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    n = len(items)
    if n <= int(limit):
        return [dict(item) for item in items]
    if limit <= 1:
        return [dict(items[0])]

    out: List[Dict[str, Any]] = []
    seen: set[int] = set()
    step = float(n - 1) / float(limit - 1)
    for i in range(int(limit)):
        idx = int(round(i * step))
        idx = max(0, min(n - 1, idx))
        if idx in seen:
            continue
        seen.add(idx)
        out.append(dict(items[idx]))

    if len(out) < int(limit):
        for idx in range(n):
            if idx in seen:
                continue
            out.append(dict(items[idx]))
            seen.add(idx)
            if len(out) >= int(limit):
                break
    return out


def _build_refine_candidates(
    *,
    top_params: Sequence[Mapping[str, Any]],
    values_map: Mapping[str, List[Any]],
    refine_neighbor_span: int,
    known: set[Tuple[Tuple[str, Any], ...]],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set(known)
    span = max(0, int(refine_neighbor_span))

    for params in top_params:
        for key, values in values_map.items():
            if not values:
                continue
            try:
                center_idx = values.index(params.get(key))
            except ValueError:
                continue
            lo = max(0, center_idx - span)
            hi = min(len(values) - 1, center_idx + span)
            for idx in range(lo, hi + 1):
                cand = dict(params)
                cand[key] = values[idx]
                cand_key = _params_key(cand)
                if cand_key in seen:
                    continue
                seen.add(cand_key)
                out.append(cand)
    return out


def _evaluate_trials(
    *,
    trials: Sequence[Mapping[str, Any]],
    trial_offset: int,
    samples: Sequence[PredictionSample],
    gt_map: Mapping[str, gpd.GeoDataFrame],
    output_dir: Path,
    base_config: Mapping[str, Any],
    iou_thr: float,
    logger: logging.Logger,
    desc: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    trial_iter = iter_progress(
        trials,
        total=len(trials),
        desc=desc,
        unit="trial",
        enabled=True,
    )
    for local_idx, params in enumerate(trial_iter, start=1):
        trial_idx = int(trial_offset + local_idx)
        trial_cfg = deep_update(dict(base_config), dict(params))
        trial_cfg["save_intermediates"] = False
        trial_cfg["export_shp"] = False

        sample_metrics = []
        sample_iter = iter_progress(
            samples,
            total=len(samples),
            desc=f"{desc}:samples",
            unit="sample",
            enabled=False,
            leave=False,
        )
        for s in sample_iter:
            res = run_postprocess_pipeline(
                extent_prob_path=s.extent_prob_path,
                boundary_prob_path=s.boundary_prob_path,
                valid_mask_path=s.valid_mask_path,
                footprint_path=s.footprint_path,
                footprint_nodata_value=s.valid_nodata_value,
                footprint_nodata_rule=s.valid_nodata_rule,
                footprint_control_band_1based=s.valid_control_band_1based,
                input_context=s.to_input_context(),
                output_dir=output_dir / "_tmp" / f"trial_{trial_idx:04d}" / s.sample_id,
                config=trial_cfg,
                save_outputs=False,
                logger=logger,
                show_progress=False,
            )
            pred_gdf = res["fields_pred"]
            gt_gdf = gt_map[s.sample_id]
            m = evaluate_polygons(
                gt_gdf=gt_gdf,
                pred_gdf=pred_gdf,
                iou_threshold=iou_thr,
                show_progress=False,
            )
            m["sample_id"] = s.sample_id
            sample_metrics.append(m)

        aggregated = aggregate_metrics(sample_metrics)
        rows.append(
            {
                "trial": trial_idx,
                "params": dict(params),
                "metrics": aggregated,
                "sample_metrics": sample_metrics,
            }
        )
    return rows


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
    sample_iter = iter_progress(
        samples,
        total=len(samples),
        desc="resolve-gt",
        unit="sample",
        enabled=True,
        leave=False,
    )
    for s in sample_iter:
        gt_path = resolve_gt_path(s.sample_id, gt_root=gt_root, gt_mode=gt_mode)
        gt_map[s.sample_id] = load_polygons(gt_path, show_progress=False)
        gt_path_map[s.sample_id] = str(gt_path)

    full_grid = build_grid(base_config)
    if not full_grid:
        raise RuntimeError("Search grid is empty")

    search_cfg = dict(base_config.get("search", {}) or {})
    strategy_cfg = dict(search_cfg.get("strategy", {}) or {})
    strategy_mode = str(strategy_cfg.get("mode", "coarse_to_fine")).strip().lower()
    max_trials_auto = int(strategy_cfg.get("max_trials_auto", 256))
    coarse_trials = int(strategy_cfg.get("coarse_trials", max_trials_auto))
    coarse_top_k = int(strategy_cfg.get("coarse_top_k", 3))
    refine_neighbor_span = int(strategy_cfg.get("refine_neighbor_span", 1))

    max_trials_total = None if max_trials is None else max(1, int(max_trials))

    iou_thr = float((base_config.get("scoring", {}) or {}).get("iou_threshold", 0.5))

    rows: List[Dict[str, Any]] = []
    coarse_count = 0
    refine_count = 0
    grid_truncated = False

    if strategy_mode == "coarse_to_fine" and len(full_grid) > 1:
        coarse_limit = max(1, min(len(full_grid), coarse_trials))
        if max_trials_total is not None:
            coarse_limit = min(coarse_limit, max_trials_total)

        coarse_grid = _sample_evenly(full_grid, coarse_limit)
        rows_coarse = _evaluate_trials(
            trials=coarse_grid,
            trial_offset=0,
            samples=samples,
            gt_map=gt_map,
            output_dir=output_dir,
            base_config=base_config,
            iou_thr=iou_thr,
            logger=log,
            desc="postprocess-coarse",
        )
        rows.extend(rows_coarse)
        coarse_count = len(rows_coarse)

        remaining = None if max_trials_total is None else max(0, max_trials_total - len(rows))
        if remaining is None or remaining > 0:
            rows_coarse_sorted = sorted(rows_coarse, key=lambda r: ranking_key(r["metrics"]), reverse=True)
            top_rows = rows_coarse_sorted[: max(1, coarse_top_k)]
            values_map = _grid_values(base_config)
            known = {_params_key(item) for item in coarse_grid}
            refine_candidates = _build_refine_candidates(
                top_params=[r["params"] for r in top_rows],
                values_map=values_map,
                refine_neighbor_span=refine_neighbor_span,
                known=known,
            )
            if remaining is not None and len(refine_candidates) > remaining:
                refine_candidates = _sample_evenly(refine_candidates, remaining)
                grid_truncated = True

            if refine_candidates:
                rows_refine = _evaluate_trials(
                    trials=refine_candidates,
                    trial_offset=len(rows),
                    samples=samples,
                    gt_map=gt_map,
                    output_dir=output_dir,
                    base_config=base_config,
                    iou_thr=iou_thr,
                    logger=log,
                    desc="postprocess-refine",
                )
                rows.extend(rows_refine)
                refine_count = len(rows_refine)
    else:
        eval_grid: List[Dict[str, Any]]
        if max_trials_total is not None:
            eval_grid = [dict(item) for item in full_grid[:max_trials_total]]
            grid_truncated = len(eval_grid) < len(full_grid)
        elif len(full_grid) > int(max_trials_auto):
            eval_grid = _sample_evenly(full_grid, int(max_trials_auto))
            grid_truncated = True
        else:
            eval_grid = [dict(item) for item in full_grid]

        rows = _evaluate_trials(
            trials=eval_grid,
            trial_offset=0,
            samples=samples,
            gt_map=gt_map,
            output_dir=output_dir,
            base_config=base_config,
            iou_thr=iou_thr,
            logger=log,
            desc="postprocess-grid",
        )
        coarse_count = len(rows)

    rows_sorted = sorted(rows, key=lambda r: ranking_key(r["metrics"]), reverse=True)
    if not rows_sorted:
        raise RuntimeError("Search produced no evaluated trials")
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
        "search_strategy": strategy_mode,
        "full_grid_trials": len(full_grid),
        "num_trials": len(rows),
        "coarse_trials": int(coarse_count),
        "refine_trials": int(refine_count),
        "grid_truncated": bool(grid_truncated),
        "max_trials_requested": max_trials_total,
        "max_trials_auto": int(max_trials_auto),
        "best_trial": best["trial"],
        "best_params_override": best["params"],
        "best_metrics": best["metrics"],
        "results": rows_sorted,
    }
    write_json(out_dir / "search_results.json", to_serializable(summary))

    return summary
