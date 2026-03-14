from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage as ndi
from shapely.geometry import shape

from .geometry_clean import count_holes


def _prepare_polygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.empty:
        return gdf
    out = gdf.copy()
    out = out[~out.geometry.isna()].copy()
    out = out[~out.geometry.is_empty].copy()
    out = out.explode(index_parts=False, ignore_index=True)
    out = out[out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    out = out.reset_index(drop=True)
    return out


def load_polygons(path: Path) -> gpd.GeoDataFrame:
    """Load GT/pred polygons from vector file or label/mask raster."""
    ext = path.suffix.lower()
    if ext in {".gpkg", ".shp", ".geojson", ".json"}:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"{path}: vector CRS is missing")
        return _prepare_polygons(gdf)

    if ext in {".tif", ".tiff"}:
        with rasterio.open(path) as ds:
            arr = ds.read(1)
            if ds.crs is None:
                raise ValueError(f"{path}: raster CRS is missing")
            transform = ds.transform
            crs = ds.crs

        if arr.dtype.kind == "f":
            mask = arr > 0.5
            labels, _ = ndi.label(mask)
        else:
            arr_int = arr.astype(np.int32)
            if int(arr_int.max()) <= 1:
                labels, _ = ndi.label(arr_int > 0)
            else:
                labels = arr_int

        records = []
        for geom_json, value in shapes(labels.astype(np.int32), mask=labels > 0, transform=transform):
            lbl = int(value)
            if lbl <= 0:
                continue
            geom = shape(geom_json)
            if geom.is_empty:
                continue
            records.append({"label_id": lbl, "geometry": geom})

        if not records:
            return gpd.GeoDataFrame({"label_id": [], "geometry": []}, geometry="geometry", crs=crs)

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        gdf = gdf.dissolve(by="label_id", as_index=False)
        gdf = gdf.explode(index_parts=False, ignore_index=True)
        return _prepare_polygons(gdf)

    raise ValueError(f"Unsupported geometry source: {path}")


def _pairwise_iou_and_merges(
    pred: gpd.GeoDataFrame,
    gt: gpd.GeoDataFrame,
    merge_overlap_ratio: float,
) -> tuple[List[Tuple[float, int, int, float]], int]:
    """Return candidate (iou,pred_idx,gt_idx,intersection_area) pairs and merge penalty count."""
    pairs: List[Tuple[float, int, int, float]] = []
    merge_count = 0

    if pred.empty or gt.empty:
        return pairs, merge_count

    gt_sindex = gt.sindex

    for p_idx, p_geom in enumerate(pred.geometry):
        cand = list(gt_sindex.intersection(p_geom.bounds))
        overlap_gt = 0

        for g_idx in cand:
            g_geom = gt.geometry.iloc[g_idx]
            inter = p_geom.intersection(g_geom)
            if inter.is_empty:
                continue
            inter_area = float(inter.area)
            if inter_area <= 0:
                continue

            gt_area = float(g_geom.area)
            if gt_area > 0 and (inter_area / gt_area) >= float(merge_overlap_ratio):
                overlap_gt += 1

            union_area = float(p_geom.area + g_geom.area - inter_area)
            if union_area <= 0:
                continue

            iou = inter_area / union_area
            if iou > 0:
                pairs.append((iou, p_idx, g_idx, inter_area))

        if overlap_gt >= 2:
            merge_count += 1

    return pairs, merge_count


def evaluate_polygons(
    gt_gdf: gpd.GeoDataFrame,
    pred_gdf: gpd.GeoDataFrame,
    iou_threshold: float = 0.5,
    merge_overlap_ratio: float = 0.2,
    area_weighted: bool = True,
) -> Dict[str, float | int]:
    """Object-level evaluation with greedy IoU matching."""
    if gt_gdf.crs is None or pred_gdf.crs is None:
        raise ValueError("Both GT and prediction GeoDataFrames must have CRS")

    gt = _prepare_polygons(gt_gdf)
    pred = _prepare_polygons(pred_gdf.to_crs(gt.crs))

    gt_count = int(len(gt))
    pred_count = int(len(pred))

    holes_penalty = count_holes(pred)
    invalid_geometries = int((~pred.geometry.is_valid).sum()) if pred_count > 0 else 0

    if gt_count == 0 and pred_count == 0:
        return {
            "gt_count": 0,
            "pred_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mean_iou_matched": 1.0,
            "merge_penalty": 0,
            "holes_penalty": 0,
            "invalid_geometries": 0,
            "area_precision": 1.0,
            "area_recall": 1.0,
            "area_f1": 1.0,
        }

    if gt_count == 0:
        fp = pred_count
        return {
            "gt_count": 0,
            "pred_count": pred_count,
            "tp": 0,
            "fp": fp,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_iou_matched": 0.0,
            "merge_penalty": 0,
            "holes_penalty": int(holes_penalty),
            "invalid_geometries": int(invalid_geometries),
            "area_precision": 0.0,
            "area_recall": 0.0,
            "area_f1": 0.0,
        }

    if pred_count == 0:
        fn = gt_count
        return {
            "gt_count": gt_count,
            "pred_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": fn,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_iou_matched": 0.0,
            "merge_penalty": 0,
            "holes_penalty": 0,
            "invalid_geometries": 0,
            "area_precision": 0.0,
            "area_recall": 0.0,
            "area_f1": 0.0,
        }

    pairs, merge_penalty = _pairwise_iou_and_merges(pred=pred, gt=gt, merge_overlap_ratio=merge_overlap_ratio)
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_pred = set()
    matched_gt = set()
    matched_ious: List[float] = []
    matched_inter_area = 0.0

    thr = float(iou_threshold)
    for iou, p_idx, g_idx, inter_area in pairs:
        if iou < thr:
            continue
        if p_idx in matched_pred or g_idx in matched_gt:
            continue
        matched_pred.add(p_idx)
        matched_gt.add(g_idx)
        matched_ious.append(float(iou))
        matched_inter_area += float(inter_area)

    tp = len(matched_ious)
    fp = pred_count - tp
    fn = gt_count - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou_matched = float(np.mean(matched_ious)) if matched_ious else 0.0

    area_precision = 0.0
    area_recall = 0.0
    area_f1 = 0.0
    if area_weighted:
        pred_area = float(pred.area.sum())
        gt_area = float(gt.area.sum())
        area_precision = matched_inter_area / pred_area if pred_area > 0 else 0.0
        area_recall = matched_inter_area / gt_area if gt_area > 0 else 0.0
        area_f1 = (
            2.0 * area_precision * area_recall / (area_precision + area_recall)
            if (area_precision + area_recall) > 0
            else 0.0
        )

    return {
        "gt_count": int(gt_count),
        "pred_count": int(pred_count),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou_matched": float(mean_iou_matched),
        "merge_penalty": int(merge_penalty),
        "holes_penalty": int(holes_penalty),
        "invalid_geometries": int(invalid_geometries),
        "area_precision": float(area_precision),
        "area_recall": float(area_recall),
        "area_f1": float(area_f1),
    }


def aggregate_metrics(rows: Sequence[Dict[str, float | int]]) -> Dict[str, float | int]:
    if not rows:
        return {
            "num_samples": 0,
            "gt_count": 0,
            "pred_count": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_iou_matched": 0.0,
            "merge_penalty": 0,
            "holes_penalty": 0,
            "invalid_geometries": 0,
            "area_precision": 0.0,
            "area_recall": 0.0,
            "area_f1": 0.0,
        }

    gt_count = int(sum(int(r["gt_count"]) for r in rows))
    pred_count = int(sum(int(r["pred_count"]) for r in rows))
    tp = int(sum(int(r["tp"]) for r in rows))
    fp = int(sum(int(r["fp"]) for r in rows))
    fn = int(sum(int(r["fn"]) for r in rows))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Weighted by matched TP count to stabilize aggregation across samples.
    iou_num = float(sum(float(r["mean_iou_matched"]) * int(r["tp"]) for r in rows))
    iou_den = float(max(1, tp))
    mean_iou_matched = iou_num / iou_den

    area_precision = float(np.mean([float(r.get("area_precision", 0.0)) for r in rows]))
    area_recall = float(np.mean([float(r.get("area_recall", 0.0)) for r in rows]))
    area_f1 = float(np.mean([float(r.get("area_f1", 0.0)) for r in rows]))

    return {
        "num_samples": int(len(rows)),
        "gt_count": gt_count,
        "pred_count": pred_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_iou_matched": float(mean_iou_matched),
        "merge_penalty": int(sum(int(r["merge_penalty"]) for r in rows)),
        "holes_penalty": int(sum(int(r["holes_penalty"]) for r in rows)),
        "invalid_geometries": int(sum(int(r["invalid_geometries"]) for r in rows)),
        "area_precision": area_precision,
        "area_recall": area_recall,
        "area_f1": area_f1,
    }


def ranking_key(metrics_row: Dict[str, float | int]) -> tuple:
    """
    Higher is better. Implements requested tie-break chain:
    1) F1
    2) mean IoU matched
    3) fewer merges
    4) fewer holes
    5) fewer invalid geometries
    """
    return (
        float(metrics_row.get("f1", 0.0)),
        float(metrics_row.get("mean_iou_matched", 0.0)),
        -int(metrics_row.get("merge_penalty", 0)),
        -int(metrics_row.get("holes_penalty", 0)),
        -int(metrics_row.get("invalid_geometries", 0)),
    )
