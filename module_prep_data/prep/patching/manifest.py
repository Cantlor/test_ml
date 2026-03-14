from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from rasterio.windows import Window


def build_patch_meta(
    ds_name: str,
    patch_id: str,
    inside_mode: str,
    feat_i: Optional[int],
    field_id: Optional[str],
    win: Window,
    valid_ratio: float,
    nodata_frac: float,
    mask_ratio: float,
    edge_ratio: float,
    skeleton_ratio: float,
    ignore_ratio: float,
    pixel_size_m: float,
    train_crop_px: int,
    pad_px: int,
    nodata_value: float,
    nodata_rule: str,
    control_band_1based: int,
    labels_written: Dict[str, bool],
) -> Dict[str, Any]:
    return {
        "dataset": ds_name,
        "patch_id": patch_id,
        "inside_mode": inside_mode,
        "feat_index": int(feat_i) if feat_i is not None else None,
        "field_id": field_id,
        "xoff": int(win.col_off),
        "yoff": int(win.row_off),
        "w": int(win.width),
        "h": int(win.height),
        "valid_ratio": float(valid_ratio),
        "nodata_frac": float(nodata_frac),
        "mask_ratio": float(mask_ratio),
        "edge_ratio": float(edge_ratio),
        "skeleton_ratio": float(skeleton_ratio),
        "ignore_ratio": float(ignore_ratio),
        "pixel_size_m": float(pixel_size_m),
        "train_crop_px": int(train_crop_px),
        "pad_px": int(pad_px),
        "nodata_policy": {
            "value": float(nodata_value),
            "rule": str(nodata_rule),
            "control_band_1based": int(control_band_1based),
        },
        "labels_written": {
            "extent": bool(labels_written.get("extent", False)),
            "extent_ig": bool(labels_written.get("extent_ig", False)),
            "boundary_raw": bool(labels_written.get("boundary_raw", False)),
            "boundary_bwbl": bool(labels_written.get("boundary_bwbl", False)),
            "valid": bool(labels_written.get("valid", False)),
        },
    }


def build_dataset_summary(
    ds_name: str,
    raster_path: str,
    vector_path: str,
    vector_layer: Optional[str],
    vector_id_field: Optional[str],
    field_id_source: str,
    written_total: int,
    written_center: int,
    written_boundary: int,
    written_negative: int,
    target_total: int,
    pos_target: int,
    center_target: int,
    boundary_target: int,
    neg_target: int,
    center_attempts: int,
    boundary_attempts: int,
    neg_attempts: int,
    rejects: Dict[str, int],
    nodata_value: float,
    nodata_rule: str,
    control_band_1based: int,
) -> Dict[str, Any]:
    return {
        "dataset": ds_name,
        "raster": str(raster_path),
        "vector": str(vector_path),
        "vector_layer": vector_layer,
        "vector_id_field": vector_id_field,
        "field_id_source": field_id_source,
        "written_total": int(written_total),
        "written_center": int(written_center),
        "written_boundary": int(written_boundary),
        "written_negative": int(written_negative),
        "targets": {
            "target_total": int(target_total),
            "pos_target": int(pos_target),
            "center_target": int(center_target),
            "boundary_target": int(boundary_target),
            "neg_target": int(neg_target),
        },
        "attempts": {
            "center": int(center_attempts),
            "boundary": int(boundary_attempts),
            "negative": int(neg_attempts),
        },
        "shortfall": {
            "center": int(max(0, center_target - written_center)),
            "boundary": int(max(0, boundary_target - written_boundary)),
            "negative": int(max(0, neg_target - written_negative)),
        },
        "rejects": dict(rejects),
        "nodata_policy": {
            "value": float(nodata_value),
            "rule": str(nodata_rule),
            "control_band_1based": int(control_band_1based),
        },
    }
