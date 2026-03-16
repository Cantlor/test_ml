from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import geopandas as gpd
import numpy as np

from .geometry_clean import clean_polygons
from .io import deep_update, ensure_dir, load_inputs, read_yaml, save_raster, to_serializable, write_json
from .metrics import evaluate_polygons, load_polygons
from .progress import bar_progress
from .raster_ops import (
    area_m2_to_px,
    build_boundary_barrier,
    build_field_mask,
    estimate_pixel_area_m2,
    log_thresholds,
    smooth_probabilities,
)
from .runtime import build_runtime_policy
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


def _empty_prediction_gdf(crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"label_id": [], "geometry": []}, geometry="geometry", crs=crs)


def run_postprocess_pipeline(
    *,
    extent_prob_path: Path,
    boundary_prob_path: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    valid_mask_path: Optional[Path] = None,
    footprint_path: Optional[Path] = None,
    footprint_nodata_value: Optional[float] = None,
    footprint_nodata_rule: str = "control-band",
    footprint_control_band_1based: int = 1,
    input_context: Optional[Mapping[str, Any]] = None,
    gt_path: Optional[Path] = None,
    save_outputs: bool = True,
    logger: Optional[logging.Logger] = None,
    show_progress: bool | None = None,
) -> Dict[str, Any]:
    """Run full raster->vector post-processing pipeline."""
    log = logger or logging.getLogger(__name__)
    sample_id = str((input_context or {}).get("sample_id") or Path(extent_prob_path).stem)
    pipeline_bar = bar_progress(
        total=6,
        desc=f"postprocess:{sample_id}",
        unit="step",
        enabled=show_progress,
        leave=False,
    )

    try:
        memory_cfg = dict(config.get("memory", {}) or {})
        prob_dtype = str(memory_cfg.get("prob_dtype", "float16"))

        bundle = load_inputs(
            extent_prob_path=extent_prob_path,
            boundary_prob_path=boundary_prob_path,
            valid_mask_path=valid_mask_path,
            footprint_path=footprint_path,
            footprint_nodata_value=footprint_nodata_value,
            footprint_nodata_rule=footprint_nodata_rule,
            footprint_control_band_1based=footprint_control_band_1based,
            prob_dtype=prob_dtype,
        )
        n_pixels = int(bundle.meta.width) * int(bundle.meta.height)
        valid_pixels = int(bundle.valid_mask.sum())

        runtime = build_runtime_policy(
            config=config,
            memory_cfg=memory_cfg,
            n_pixels=n_pixels,
            valid_pixels=valid_pixels,
        )
        for warn in runtime.warnings:
            log.warning("Runtime policy: %s", warn)
        pipeline_bar.update(1)

        pixel_area_m2 = estimate_pixel_area_m2(
            crs=bundle.meta.crs,
            transform=bundle.meta.transform,
            width=bundle.meta.width,
            height=bundle.meta.height,
        )

        extent_thr = _get_float(config, "extent_thr", 0.5)
        boundary_thr = _get_float(config, "boundary_thr", 0.5)
        gaussian_sigma_px = float(runtime.gaussian_sigma_px_effective)
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
        boundary_weight = _get_float(config, "boundary_weight", 2.5)
        use_watershed = bool(runtime.use_watershed)
        sobel_weight = float(runtime.sobel_weight_effective)
        smooth_dtype = str(runtime.smooth_dtype)
        clean_labels_mode = str(runtime.clean_labels_mode)
        clean_labels_max_exact_hole_labels = _get_int(memory_cfg, "clean_labels_max_exact_hole_labels", 512)
        clean_labels_max_exact_merge_regions = _get_int(memory_cfg, "clean_labels_max_exact_merge_regions", 2048)

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

        early_exit_reason: Optional[str] = None

        if valid_pixels <= 0:
            extent_smooth = bundle.extent_prob.astype(np.dtype(smooth_dtype), copy=False)
            boundary_smooth = bundle.boundary_prob.astype(np.dtype(smooth_dtype), copy=False)
            field_mask = np.zeros((bundle.meta.height, bundle.meta.width), dtype=bool)
            boundary_barrier = np.zeros((bundle.meta.height, bundle.meta.width), dtype=bool)
            seeds = np.zeros((bundle.meta.height, bundle.meta.width), dtype=np.int32)
            labels = np.zeros((bundle.meta.height, bundle.meta.width), dtype=np.int32)
            labels_stats_obj = {"num_labels": 0, "max_label": 0, "num_pixels_fg": 0}
            raw_gdf = _empty_prediction_gdf(bundle.meta.crs)
            early_exit_reason = "all_invalid"
        elif float(bundle.extent_prob.max()) < float(extent_thr):
            extent_smooth = bundle.extent_prob.astype(np.dtype(smooth_dtype), copy=False)
            boundary_smooth = bundle.boundary_prob.astype(np.dtype(smooth_dtype), copy=False)
            field_mask = np.zeros((bundle.meta.height, bundle.meta.width), dtype=bool)
            boundary_barrier = np.zeros((bundle.meta.height, bundle.meta.width), dtype=bool)
            seeds = np.zeros((bundle.meta.height, bundle.meta.width), dtype=np.int32)
            labels = np.zeros((bundle.meta.height, bundle.meta.width), dtype=np.int32)
            labels_stats_obj = {"num_labels": 0, "max_label": 0, "num_pixels_fg": 0}
            raw_gdf = _empty_prediction_gdf(bundle.meta.crs)
            early_exit_reason = "no_extent_above_threshold"
        else:
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
            if not np.any(field_mask):
                boundary_barrier = np.zeros(field_mask.shape, dtype=bool)
                seeds = np.zeros(field_mask.shape, dtype=np.int32)
                labels = np.zeros(field_mask.shape, dtype=np.int32)
                labels_stats_obj = {"num_labels": 0, "max_label": 0, "num_pixels_fg": 0}
                raw_gdf = _empty_prediction_gdf(bundle.meta.crs)
                early_exit_reason = "empty_field_mask"
            else:
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
                    mode=clean_labels_mode,
                    max_exact_hole_labels=clean_labels_max_exact_hole_labels,
                    max_exact_merge_regions=clean_labels_max_exact_merge_regions,
                    show_progress=show_progress,
                )
                labels_stats_obj = labels_stats(labels)
                raw_gdf = labels_to_geodataframe(
                    labels=labels,
                    transform=bundle.meta.transform,
                    crs=bundle.meta.crs,
                    show_progress=show_progress,
                )
        pipeline_bar.update(1)

        if early_exit_reason is not None:
            use_watershed = False

        valid_geom = None
        if clip_to_valid and valid_pixels > 0 and valid_pixels < n_pixels:
            valid_geom = valid_mask_to_geometry(
                valid_mask=bundle.valid_mask,
                transform=bundle.meta.transform,
                show_progress=show_progress,
            )
        if clip_to_valid and valid_geom is not None and not raw_gdf.empty:
            raw_gdf = clip_geodataframe_to_geom(raw_gdf, valid_geom)

        final_gdf = clean_polygons(
            raw_gdf=raw_gdf,
            min_area_m2=min_area_m2,
            simplify_m=simplify_m,
            remove_holes=remove_holes,
            clip_geom=None,
            straighten_cfg=straighten_cfg,
        )
        pipeline_bar.update(1)

        metrics_postproc: Optional[Dict[str, Any]] = None
        if gt_path is not None:
            gt_gdf = load_polygons(gt_path.resolve(), show_progress=show_progress)
            iou_thr = float((config.get("scoring", {}) or {}).get("iou_threshold", 0.5))
            metrics_postproc = evaluate_polygons(
                gt_gdf=gt_gdf,
                pred_gdf=final_gdf,
                iou_threshold=iou_thr,
                show_progress=show_progress,
            )
        pipeline_bar.update(1)

        outputs: Dict[str, Any] = {
            "pixel_area_m2": pixel_area_m2,
            "labels_stats": labels_stats_obj,
            "fields_pred_raw": raw_gdf,
            "fields_pred": final_gdf,
            "metrics_postproc": metrics_postproc,
            "valid_source": bundle.valid_source,
            "valid_context": bundle.valid_context,
            "memory_runtime": deep_update(
                runtime.to_dict(),
                {
                    "n_pixels": n_pixels,
                    "valid_pixels": valid_pixels,
                    "valid_fraction": (float(valid_pixels) / float(max(1, n_pixels))),
                    "prob_dtype": prob_dtype,
                    "smooth_dtype": smooth_dtype,
                    "gaussian_sigma_px_effective": gaussian_sigma_px,
                    "use_watershed_effective": use_watershed,
                    "clean_labels_mode": clean_labels_mode,
                    "early_exit_reason": early_exit_reason,
                },
            ),
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
                    "valid_source": bundle.valid_source,
                    "valid_context": to_serializable(bundle.valid_context),
                    "input_context": to_serializable(dict(input_context or {})),
                    "pixel_area_m2": pixel_area_m2,
                    "config": to_serializable(dict(config)),
                    "labels_stats": labels_stats_obj,
                    "memory_runtime": to_serializable(outputs["memory_runtime"]),
                },
            )

            if metrics_postproc is not None:
                write_json(out_dir / "metrics_postproc.json", metrics_postproc)
                outputs["metrics_postproc_json"] = str(out_dir / "metrics_postproc.json")

            manifest_outputs = {
                key: value
                for key, value in outputs.items()
                if isinstance(value, str) and key.endswith(("_tif", "_gpkg", "_shp", "_json"))
            }
            postprocess_manifest_path = out_dir / "postprocess_manifest.json"
            write_json(
                postprocess_manifest_path,
                {
                    "schema_version": "1.0",
                    "extent_prob_path": str(extent_prob_path.resolve()),
                    "boundary_prob_path": str(boundary_prob_path.resolve()),
                    "valid_mask_path": str(valid_mask_path.resolve()) if valid_mask_path is not None else None,
                    "footprint_path": str(footprint_path.resolve()) if footprint_path is not None else None,
                    "valid_source": bundle.valid_source,
                    "valid_context": to_serializable(bundle.valid_context),
                    "input_context": to_serializable(dict(input_context or {})),
                    "labels_stats": labels_stats_obj,
                    "pixel_area_m2": pixel_area_m2,
                    "memory_runtime": to_serializable(outputs["memory_runtime"]),
                    "config": to_serializable(dict(config)),
                    "outputs": manifest_outputs,
                    "metrics_postproc": to_serializable(metrics_postproc),
                },
            )
            outputs["postprocess_manifest_json"] = str(postprocess_manifest_path)

        pipeline_bar.update(1)
        pipeline_bar.update(1)
        return outputs
    finally:
        pipeline_bar.close()
