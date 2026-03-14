from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import rasterio

from .geometry_clean import clean_polygons
from .io import deep_update, ensure_dir, load_inputs, read_yaml, save_raster, to_serializable, write_json
from .metrics import evaluate_polygons, load_polygons
from .raster_ops import (
    area_m2_to_px,
    build_boundary_barrier,
    build_field_mask,
    estimate_pixel_area_m2,
    log_thresholds,
    smooth_probabilities,
)
from .seeds import build_markers
from .separation import clean_labels, labels_stats, split_fields
from .vectorize import (
    clip_geodataframe_to_geom,
    labels_to_geodataframe,
    save_geodataframe,
    valid_mask_to_geometry,
)


def default_config_path() -> Path:
    return (Path(__file__).resolve().parents[1] / "configs" / "postprocess_config.yaml").resolve()


def load_config(config_path: Optional[Path] = None, override_path: Optional[Path] = None, override: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    cfg_path = (config_path or default_config_path()).resolve()
    cfg = read_yaml(cfg_path)

    if override_path is not None:
        cfg = deep_update(cfg, read_yaml(override_path.resolve()))

    if override is not None:
        cfg = deep_update(cfg, dict(override))

    return cfg


def _get_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return float(default)


def _get_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return int(default)


def run_postprocess_pipeline(
    *,
    extent_prob_path: Path,
    boundary_prob_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    valid_mask_path: Optional[Path] = None,
    footprint_path: Optional[Path] = None,
    gt_path: Optional[Path] = None,
    save_outputs: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Run full raster->vector post-processing pipeline."""
    log = logger or logging.getLogger(__name__)

    memory_cfg = dict(config.get("memory", {}) or {})
    prob_dtype = str(memory_cfg.get("prob_dtype", "float16"))
    auto_disable_watershed = bool(memory_cfg.get("auto_disable_watershed", True))
    force_no_watershed = bool(memory_cfg.get("force_no_watershed", False))
    max_pixels_for_watershed = int(memory_cfg.get("max_pixels_for_watershed", 50_000_000))
    warn_pixels_threshold = int(memory_cfg.get("warn_pixels_threshold", 30_000_000))
    auto_disable_gaussian_large = bool(memory_cfg.get("auto_disable_gaussian_large", True))
    max_pixels_for_gaussian = int(memory_cfg.get("max_pixels_for_gaussian", max_pixels_for_watershed))
    sobel_weight_default = _get_float(config, "sobel_weight", 0.0)
    sobel_weight_large = _get_float(memory_cfg, "sobel_weight_large", 0.0)

    with rasterio.open(extent_prob_path) as ref_ds:
        n_pixels = int(ref_ds.width) * int(ref_ds.height)
    if n_pixels >= warn_pixels_threshold:
        log.warning(
            "Large raster detected: %d pixels (%.2f MP). Using memory-safe settings where needed.",
            n_pixels,
            n_pixels / 1_000_000.0,
        )

    bundle = load_inputs(
        extent_prob_path=extent_prob_path,
        boundary_prob_path=boundary_prob_path,
        valid_mask_path=valid_mask_path,
        footprint_path=footprint_path,
        prob_dtype=prob_dtype,
    )

    pixel_area_m2 = estimate_pixel_area_m2(
        crs=bundle.meta.crs,
        transform=bundle.meta.transform,
        width=bundle.meta.width,
        height=bundle.meta.height,
    )

    extent_thr = _get_float(config, "extent_thr", 0.5)
    boundary_thr = _get_float(config, "boundary_thr", 0.5)
    gaussian_sigma_px = _get_float(config, "gaussian_sigma_px", 1.0)
    if auto_disable_gaussian_large and (n_pixels > max_pixels_for_gaussian):
        if gaussian_sigma_px > 0:
            log.warning(
                "Gaussian smoothing disabled by memory safety: pixels=%d > max_pixels_for_gaussian=%d",
                n_pixels,
                max_pixels_for_gaussian,
            )
        gaussian_sigma_px = 0.0
    boundary_dilate_px = _get_int(config, "boundary_dilate_px", 1)

    fill_holes_max_area_m2 = _get_float(config, "fill_holes_max_area_m2", 50.0)
    small_region_max_area_m2 = _get_float(config, "small_region_max_area_m2", 30.0)
    min_area_m2 = _get_float(config, "min_area_m2", 100.0)

    remove_small_objects_m2 = _get_float(config, "remove_small_objects_m2", small_region_max_area_m2)
    opening_px = _get_int(config, "opening_px", 0)
    closing_px = _get_int(config, "closing_px", 0)

    seed_min_distance_px = _get_int(config, "seed_min_distance_px", 6)
    seed_hmax = _get_float(config, "seed_hmax", 1.0)
    marker_erode_px = _get_int(config, "marker_erode_px", 1)
    use_watershed_cfg = bool(config.get("use_watershed", True))
    boundary_weight = _get_float(config, "boundary_weight", 2.5)

    disable_watershed_large = auto_disable_watershed and (n_pixels > max_pixels_for_watershed)
    use_watershed = bool(use_watershed_cfg and not force_no_watershed and not disable_watershed_large)
    if use_watershed_cfg and not use_watershed:
        log.warning(
            "Watershed disabled by memory safety: pixels=%d > max_pixels_for_watershed=%d",
            n_pixels,
            max_pixels_for_watershed,
        )

    sobel_weight = float(sobel_weight_default)
    if disable_watershed_large:
        sobel_weight = float(sobel_weight_large)

    smooth_dtype = str(memory_cfg.get("smooth_dtype", prob_dtype if disable_watershed_large else "float32"))

    simplify_m = _get_float(config, "simplify_m", 1.0)
    remove_holes = bool(config.get("remove_holes", True))
    clip_to_valid = bool(config.get("clip_to_valid", True))
    straighten_cfg = dict(config.get("straighten", {}) or {})

    fill_holes_px = area_m2_to_px(fill_holes_max_area_m2, pixel_area_m2)
    small_region_px = area_m2_to_px(small_region_max_area_m2, pixel_area_m2)
    min_region_px = area_m2_to_px(min_area_m2, pixel_area_m2)
    remove_small_objects_px = area_m2_to_px(remove_small_objects_m2, pixel_area_m2)

    log_thresholds(
        pixel_area_m2,
        fill_holes_max_area_m2=fill_holes_max_area_m2,
        small_region_max_area_m2=small_region_max_area_m2,
        min_area_m2=min_area_m2,
        remove_small_objects_m2=remove_small_objects_m2,
    )

    extent_smooth, boundary_smooth = smooth_probabilities(
        extent_prob=bundle.extent_prob,
        boundary_prob=bundle.boundary_prob,
        sigma_px=gaussian_sigma_px,
        valid_mask=bundle.valid_mask,
        output_dtype=smooth_dtype,
    )

    field_mask = build_field_mask(
        extent_smooth=extent_smooth,
        valid_mask=bundle.valid_mask,
        extent_thr=extent_thr,
        remove_small_objects_px=remove_small_objects_px,
        fill_holes_px=fill_holes_px,
        opening_px=opening_px,
        closing_px=closing_px,
    )

    boundary_barrier = build_boundary_barrier(
        boundary_smooth=boundary_smooth,
        valid_mask=bundle.valid_mask,
        boundary_thr=boundary_thr,
        boundary_dilate_px=boundary_dilate_px,
    )

    if use_watershed:
        seeds, _distance = build_markers(
            field_mask=field_mask,
            boundary_barrier=boundary_barrier,
            seed_min_distance_px=seed_min_distance_px,
            seed_hmax=seed_hmax,
            marker_erode_px=marker_erode_px,
        )
        split_mask = field_mask
    else:
        seeds = np.zeros(field_mask.shape, dtype=np.int32)
        # For no-watershed fallback, enforce boundary split directly in binary mask.
        split_mask = field_mask & (~boundary_barrier)

    labels_raw = split_fields(
        field_mask=split_mask,
        extent_smooth=extent_smooth,
        boundary_smooth=boundary_smooth,
        markers=seeds,
        use_watershed=use_watershed,
        boundary_weight=boundary_weight,
        sobel_weight=sobel_weight,
    )

    labels = clean_labels(
        labels=labels_raw,
        min_region_area_px=min_region_px,
        fill_holes_max_area_px=fill_holes_px,
        small_region_max_area_px=small_region_px,
        valid_mask=bundle.valid_mask,
    )

    raw_gdf = labels_to_geodataframe(labels=labels, transform=bundle.meta.transform, crs=bundle.meta.crs)

    valid_geom = valid_mask_to_geometry(valid_mask=bundle.valid_mask, transform=bundle.meta.transform)
    if clip_to_valid and valid_geom is not None and not raw_gdf.empty:
        raw_gdf = clip_geodataframe_to_geom(raw_gdf, valid_geom)

    final_gdf = clean_polygons(
        raw_gdf=raw_gdf,
        min_area_m2=min_area_m2,
        simplify_m=simplify_m,
        remove_holes=remove_holes,
        clip_geom=valid_geom if clip_to_valid else None,
        straighten_cfg=straighten_cfg,
    )

    metrics_postproc: Optional[Dict[str, Any]] = None
    if gt_path is not None:
        gt_gdf = load_polygons(gt_path.resolve())
        iou_thr = float((config.get("scoring", {}) or {}).get("iou_threshold", 0.5))
        metrics_postproc = evaluate_polygons(gt_gdf=gt_gdf, pred_gdf=final_gdf, iou_threshold=iou_thr)

    outputs: Dict[str, Any] = {
        "pixel_area_m2": pixel_area_m2,
        "labels_stats": labels_stats(labels),
        "fields_pred_raw": raw_gdf,
        "fields_pred": final_gdf,
        "metrics_postproc": metrics_postproc,
        "memory_runtime": {
            "n_pixels": n_pixels,
            "prob_dtype": prob_dtype,
            "smooth_dtype": smooth_dtype,
            "gaussian_sigma_px_effective": gaussian_sigma_px,
            "use_watershed_effective": use_watershed,
            "max_pixels_for_watershed": max_pixels_for_watershed,
            "max_pixels_for_gaussian": max_pixels_for_gaussian,
        },
    }

    if save_outputs:
        out_dir = ensure_dir(output_dir.resolve())

        save_intermediates = bool(config.get("save_intermediates", True))
        if save_intermediates:
            outputs["extent_smooth_tif"] = str(
                save_raster(out_dir / "extent_smooth.tif", extent_smooth, bundle.meta, dtype="float32", nodata=0.0)
            )
            outputs["boundary_smooth_tif"] = str(
                save_raster(out_dir / "boundary_smooth.tif", boundary_smooth, bundle.meta, dtype="float32", nodata=0.0)
            )
            outputs["field_mask_tif"] = str(
                save_raster(out_dir / "field_mask.tif", field_mask.astype(np.uint8), bundle.meta, dtype="uint8", nodata=0)
            )
            outputs["boundary_barrier_tif"] = str(
                save_raster(
                    out_dir / "boundary_barrier.tif",
                    boundary_barrier.astype(np.uint8),
                    bundle.meta,
                    dtype="uint8",
                    nodata=0,
                )
            )
            outputs["seeds_tif"] = str(
                save_raster(out_dir / "seeds.tif", seeds.astype(np.int32), bundle.meta, dtype="int32", nodata=0)
            )
            outputs["labels_tif"] = str(
                save_raster(out_dir / "labels.tif", labels.astype(np.int32), bundle.meta, dtype="int32", nodata=0)
            )

        raw_path = out_dir / "fields_pred_raw.gpkg"
        final_path = out_dir / "fields_pred.gpkg"
        save_geodataframe(raw_gdf, raw_path)
        save_geodataframe(final_gdf, final_path)
        outputs["fields_pred_raw_gpkg"] = str(raw_path)
        outputs["fields_pred_gpkg"] = str(final_path)

        if bool(config.get("export_shp", False)):
            shp_path = out_dir / "fields_pred.shp"
            save_geodataframe(final_gdf, shp_path)
            outputs["fields_pred_shp"] = str(shp_path)

        write_json(
            out_dir / "params_used.json",
            {
                "extent_prob_path": str(extent_prob_path.resolve()),
                "boundary_prob_path": str(boundary_prob_path.resolve()),
                "valid_mask_path": str(valid_mask_path.resolve()) if valid_mask_path is not None else None,
                "footprint_path": str(footprint_path.resolve()) if footprint_path is not None else None,
                "pixel_area_m2": pixel_area_m2,
                "config": to_serializable(dict(config)),
                "labels_stats": labels_stats(labels),
                "memory_runtime": to_serializable(outputs["memory_runtime"]),
            },
        )

        if metrics_postproc is not None:
            write_json(out_dir / "metrics_postproc.json", metrics_postproc)
            outputs["metrics_postproc_json"] = str(out_dir / "metrics_postproc.json")

    return outputs
