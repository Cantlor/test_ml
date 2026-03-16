from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window

from .nodata import valid_mask_from_chip


def stable_dataset_seed(base_seed: int, ds_name: str) -> int:
    h = hashlib.sha256(ds_name.encode("utf-8")).digest()
    ds_part = int.from_bytes(h[:8], "little", signed=False)
    return int((int(base_seed) + ds_part) % (2**63 - 1))


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def _effective_sampling_stride_px(
    *,
    patch_size_px: int,
    train_crop_px: int,
    capacity_overlap_factor: float,
) -> float:
    """
    Estimate effective center spacing for sampling-capacity (not strict grid tiling).
    - lower stride => higher overlap => higher sampling capacity
    - train_crop provides a practical lower bound of useful shift diversity
    """
    patch_size = float(max(1, patch_size_px))
    crop = float(train_crop_px) if int(train_crop_px) > 0 else float(patch_size * 0.50)

    overlap_ratio = _clamp(float(capacity_overlap_factor), 0.05, 1.0)
    stride_from_overlap = float(patch_size * overlap_ratio)

    stride = float(max(crop, stride_from_overlap))
    stride = _clamp(stride, 1.0, patch_size)
    return stride


def compute_patch_targets(
    total_target: int,
    negatives_ratio: float,
    center_weight: float,
    boundary_weight: float,
) -> Dict[str, int]:
    total = int(max(0, total_target))
    neg_share = _clamp(float(negatives_ratio), 0.0, 1.0)
    pos_target = int(round(total * (1.0 - neg_share)))
    neg_target = int(max(0, total - pos_target))

    w_sum = float(max(1e-9, center_weight + boundary_weight))
    center_target = int(round(pos_target * (float(center_weight) / w_sum)))
    boundary_target = int(max(0, pos_target - center_target))

    return {
        "target_total": int(total),
        "pos_target": int(pos_target),
        "center_target": int(center_target),
        "boundary_target": int(boundary_target),
        "neg_target": int(neg_target),
    }


def _type_shares(negatives_ratio: float, center_weight: float, boundary_weight: float) -> Dict[str, float]:
    neg_share = _clamp(float(negatives_ratio), 0.0, 1.0)
    pos_share = float(max(0.0, 1.0 - neg_share))

    w_sum = float(center_weight + boundary_weight)
    if w_sum <= 1e-9:
        center_share = pos_share
        boundary_share = 0.0
    else:
        center_share = pos_share * float(center_weight / w_sum)
        boundary_share = pos_share - center_share

    return {
        "center": float(center_share),
        "boundary": float(boundary_share),
        "negative": float(neg_share),
    }


def derive_auto_patch_budget_from_inputs(
    *,
    patch_size_px: int,
    train_crop_px: int,
    negatives_ratio: float,
    center_weight: float,
    boundary_weight: float,
    capacity_overlap_factor: float,
    boundary_pixels_per_patch: float,
    min_valid_patches_to_keep: int,
    min_patches_per_dataset: int,
    max_patches_per_dataset: int,
    raster_area_pixels: float,
    valid_area_ratio: float,
    field_area_pixels: float,
    boundary_length_pixels: float,
) -> Dict[str, Any]:
    patch_size = int(max(1, patch_size_px))
    patch_area = float(patch_size * patch_size)
    train_crop = int(max(1, min(int(train_crop_px), patch_size)))

    raster_area = float(max(0.0, raster_area_pixels))
    valid_ratio = _clamp(float(valid_area_ratio), 0.0, 1.0)
    valid_area = float(raster_area * valid_ratio)

    field_area = float(max(0.0, field_area_pixels))
    boundary_len = float(max(0.0, boundary_length_pixels))
    non_field_valid_area = float(max(0.0, valid_area - field_area))

    overlap = _clamp(float(capacity_overlap_factor), 0.05, 1.0)
    effective_stride_px = _effective_sampling_stride_px(
        patch_size_px=patch_size,
        train_crop_px=train_crop,
        capacity_overlap_factor=overlap,
    )
    sampling_cell_area = float(effective_stride_px * effective_stride_px)

    boundary_span = float(boundary_pixels_per_patch)
    if boundary_span <= 0:
        boundary_span = float(patch_size) * 1.25

    geom_capacity = float(raster_area / sampling_cell_area) if sampling_cell_area > 0 else 0.0
    valid_capacity = float(valid_area / sampling_cell_area) if sampling_cell_area > 0 else 0.0
    field_capacity = float(field_area / sampling_cell_area) if sampling_cell_area > 0 else 0.0
    boundary_capacity_raw = float(boundary_len / max(1.0, boundary_span))
    negative_capacity_raw = float(non_field_valid_area / sampling_cell_area) if sampling_cell_area > 0 else 0.0

    est_center_capacity = int(max(0, math.floor(min(field_capacity, valid_capacity))))
    est_boundary_capacity = int(max(0, math.floor(min(boundary_capacity_raw, valid_capacity))))
    est_negative_capacity = int(max(0, math.floor(min(negative_capacity_raw, valid_capacity))))
    area_limited_capacity = int(max(0, math.floor(min(geom_capacity, valid_capacity))))

    shares = _type_shares(
        negatives_ratio=negatives_ratio,
        center_weight=center_weight,
        boundary_weight=boundary_weight,
    )

    type_bounds: Dict[str, float] = {}
    for k, share in shares.items():
        if share <= 1e-9:
            continue
        cap = {
            "center": est_center_capacity,
            "boundary": est_boundary_capacity,
            "negative": est_negative_capacity,
        }[k]
        if cap <= 0:
            type_bounds[k] = 0.0
        else:
            type_bounds[k] = float(cap / share)

    type_limited_capacity = float("inf")
    if type_bounds:
        type_limited_capacity = min(type_bounds.values())

    estimated_total_capacity = int(max(0, math.floor(min(float(area_limited_capacity), type_limited_capacity))))

    estimated_total_target = int(estimated_total_capacity)
    max_cap = int(max_patches_per_dataset)
    if max_cap > 0:
        estimated_total_target = int(min(estimated_total_target, max_cap))

    keep_threshold = int(max(0, max(int(min_valid_patches_to_keep), int(min_patches_per_dataset))))
    skipped = bool(estimated_total_capacity < keep_threshold)
    skip_reason = None
    if skipped:
        skip_reason = (
            f"estimated_total_capacity={estimated_total_capacity} "
            f"< min_valid_patches_to_keep={keep_threshold}"
        )
        estimated_total_target = 0

    target_counts = compute_patch_targets(
        total_target=int(estimated_total_target),
        negatives_ratio=negatives_ratio,
        center_weight=center_weight,
        boundary_weight=boundary_weight,
    )

    bounds_for_reason: Dict[str, float] = {"area": float(area_limited_capacity)}
    bounds_for_reason.update(type_bounds)
    min_bound = min(bounds_for_reason.values()) if bounds_for_reason else 0.0
    limiters = sorted([k for k, v in bounds_for_reason.items() if abs(v - min_bound) < 1e-9])

    return {
        "patch_size_px": int(patch_size),
        "train_crop_px": int(train_crop),
        "patch_budget_mode": "auto",
        "capacity_overlap_factor": float(overlap),
        "effective_stride_px": float(effective_stride_px),
        "sampling_cell_area_pixels": float(sampling_cell_area),
        "boundary_pixels_per_patch": float(boundary_span),
        "raster_area_pixels": float(raster_area),
        "patch_area_pixels": float(patch_area),
        "valid_area_ratio": float(valid_ratio),
        "valid_area_pixels": float(valid_area),
        "field_area_pixels": float(field_area),
        "boundary_length_pixels": float(boundary_len),
        "non_field_valid_area_pixels": float(non_field_valid_area),
        "estimated_total_capacity": int(estimated_total_capacity),
        "estimated_center_capacity": int(est_center_capacity),
        "estimated_boundary_capacity": int(est_boundary_capacity),
        "estimated_negative_capacity": int(est_negative_capacity),
        "estimated_total_target": int(target_counts["target_total"]),
        "estimated_center_target": int(target_counts["center_target"]),
        "estimated_boundary_target": int(target_counts["boundary_target"]),
        "estimated_negative_target": int(target_counts["neg_target"]),
        "skipped_due_to_low_capacity": bool(skipped),
        "skip_reason": skip_reason,
        "capacity_limiters": limiters,
        "capacity_bounds": {k: float(v) for k, v in bounds_for_reason.items()},
        "shares": {k: float(v) for k, v in shares.items()},
    }


def _estimate_valid_area_ratio_sampled(
    *,
    ds: rasterio.DatasetReader,
    patch_size_px: int,
    sample_windows: int,
    nodata_value: float,
    nodata_rule: str,
    control_band_1based: int,
    seed: int,
) -> float:
    if ds.width <= 0 or ds.height <= 0:
        return 0.0

    ws = int(min(max(64, patch_size_px), ds.width, ds.height))
    if ws <= 0:
        return 0.0

    rng = np.random.default_rng(int(seed))
    n = int(max(1, sample_windows))

    max_x = int(ds.width - ws)
    max_y = int(ds.height - ws)

    valid_px = 0
    total_px = 0
    for _ in range(n):
        if max_x <= 0:
            xoff = 0
        else:
            xoff = int(rng.integers(0, max_x + 1))
        if max_y <= 0:
            yoff = 0
        else:
            yoff = int(rng.integers(0, max_y + 1))

        win = Window(xoff, yoff, ws, ws)
        arr = ds.read(window=win)
        valid = valid_mask_from_chip(
            chip=arr,
            nodata_value=float(nodata_value),
            nodata_rule=str(nodata_rule),
            control_band_1based=int(control_band_1based),
        )
        valid_px += int(valid.sum())
        total_px += int(valid.size)

    if total_px <= 0:
        return 0.0
    return float(valid_px / total_px)


def estimate_auto_patch_budget(
    *,
    ds: rasterio.DatasetReader,
    gdf: gpd.GeoDataFrame,
    patch_size_px: int,
    train_crop_px: int,
    negatives_ratio: float,
    center_weight: float,
    boundary_weight: float,
    capacity_overlap_factor: float,
    boundary_pixels_per_patch: float,
    min_valid_patches_to_keep: int,
    min_patches_per_dataset: int,
    max_patches_per_dataset: int,
    valid_ratio_sample_windows: int,
    nodata_value: float,
    nodata_rule: str,
    control_band_1based: int,
    seed: int,
) -> Dict[str, Any]:
    raster_area_pixels = float(max(0, int(ds.width) * int(ds.height)))

    valid_area_ratio = _estimate_valid_area_ratio_sampled(
        ds=ds,
        patch_size_px=int(patch_size_px),
        sample_windows=int(valid_ratio_sample_windows),
        nodata_value=float(nodata_value),
        nodata_rule=str(nodata_rule),
        control_band_1based=int(control_band_1based),
        seed=int(seed),
    )

    det = float(abs(ds.transform.a * ds.transform.e - ds.transform.b * ds.transform.d))
    pixel_area_units = det if det > 1e-12 else 1.0
    pixel_scale_units = math.sqrt(pixel_area_units)

    field_area_units = float(gdf.geometry.area.sum()) if len(gdf) > 0 else 0.0
    boundary_len_units = float(gdf.boundary.length.sum()) if len(gdf) > 0 else 0.0

    field_area_pixels = float(field_area_units / pixel_area_units)
    boundary_length_pixels = float(boundary_len_units / max(1e-9, pixel_scale_units))

    out = derive_auto_patch_budget_from_inputs(
        patch_size_px=int(patch_size_px),
        train_crop_px=int(train_crop_px),
        negatives_ratio=float(negatives_ratio),
        center_weight=float(center_weight),
        boundary_weight=float(boundary_weight),
        capacity_overlap_factor=float(capacity_overlap_factor),
        boundary_pixels_per_patch=float(boundary_pixels_per_patch),
        min_valid_patches_to_keep=int(min_valid_patches_to_keep),
        min_patches_per_dataset=int(min_patches_per_dataset),
        max_patches_per_dataset=int(max_patches_per_dataset),
        raster_area_pixels=float(raster_area_pixels),
        valid_area_ratio=float(valid_area_ratio),
        field_area_pixels=float(field_area_pixels),
        boundary_length_pixels=float(boundary_length_pixels),
    )
    out["valid_ratio_sample_windows"] = int(max(1, valid_ratio_sample_windows))
    return out


def load_vector_for_budget(
    *,
    vector_path: str | Path,
    raster_crs,
    vector_layer: Optional[str] = None,
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(str(vector_path), layer=vector_layer) if vector_layer else gpd.read_file(str(vector_path))
    if gdf.empty:
        raise RuntimeError("Vector is empty")
    if gdf.crs is None:
        raise RuntimeError("Vector CRS missing")

    gdf = gdf.to_crs(raster_crs)
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        raise RuntimeError("No polygon features left after CRS/geometry filtering")
    return gdf


def estimate_auto_patch_budget_for_paths(
    *,
    raster_path: str | Path,
    vector_path: str | Path,
    vector_layer: Optional[str],
    patch_size_px: int,
    train_crop_px: int,
    negatives_ratio: float,
    center_weight: float,
    boundary_weight: float,
    capacity_overlap_factor: float,
    boundary_pixels_per_patch: float,
    min_valid_patches_to_keep: int,
    min_patches_per_dataset: int,
    max_patches_per_dataset: int,
    valid_ratio_sample_windows: int,
    nodata_value: float,
    nodata_rule: str,
    control_band_1based: int,
    seed: int,
) -> Dict[str, Any]:
    with rasterio.open(str(raster_path)) as ds:
        if ds.crs is None:
            raise RuntimeError("Raster CRS missing")
        gdf = load_vector_for_budget(vector_path=vector_path, raster_crs=ds.crs, vector_layer=vector_layer)
        return estimate_auto_patch_budget(
            ds=ds,
            gdf=gdf,
            patch_size_px=int(patch_size_px),
            train_crop_px=int(train_crop_px),
            negatives_ratio=float(negatives_ratio),
            center_weight=float(center_weight),
            boundary_weight=float(boundary_weight),
            capacity_overlap_factor=float(capacity_overlap_factor),
            boundary_pixels_per_patch=float(boundary_pixels_per_patch),
            min_valid_patches_to_keep=int(min_valid_patches_to_keep),
            min_patches_per_dataset=int(min_patches_per_dataset),
            max_patches_per_dataset=int(max_patches_per_dataset),
            valid_ratio_sample_windows=int(valid_ratio_sample_windows),
            nodata_value=float(nodata_value),
            nodata_rule=str(nodata_rule),
            control_band_1based=int(control_band_1based),
            seed=int(seed),
        )
