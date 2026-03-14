from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box
from skimage.morphology import skeletonize


def drop_holes_geom(geom):
    if geom is None or geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)

    if isinstance(geom, MultiPolygon):
        polys = [Polygon(p.exterior) for p in geom.geoms if p is not None and not p.is_empty]
        if not polys:
            return GeometryCollection()
        return MultiPolygon(polys)

    if isinstance(geom, GeometryCollection):
        out = []
        for g in geom.geoms:
            gg = drop_holes_geom(g)
            if gg is not None and not gg.is_empty:
                out.append(gg)
        return GeometryCollection(out)

    return geom


def safe_window_centered(col: int, row: int, w: int, h: int, width: int, height: int) -> Optional[Tuple[int, int]]:
    xoff = int(col - w // 2)
    yoff = int(row - h // 2)
    if xoff < 0 or yoff < 0:
        return None
    if xoff + w > width or yoff + h > height:
        return None
    return xoff, yoff


def window_bounds(ds: rasterio.DatasetReader, win: Window) -> Tuple[float, float, float, float]:
    b = rasterio.windows.bounds(win, ds.transform)
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def extent_and_boundaries_for_window(
    ds: rasterio.DatasetReader,
    gdf: gpd.GeoDataFrame,
    sindex,
    win: Window,
    include_holes: bool,
    ignore_enabled: bool,
    ignore_value: int,
    ignore_apply_to_extent: bool,
    ignore_radius_px: int,
    pad_px: int,
    bwbl_buffer_px: int,
    bwbl_background_value: int,
    bwbl_skeleton_value: int,
    bwbl_buffer_value: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Returns:
      extent (uint8 0/1),
      extent_ig (uint8 0/1/255),
      boundary_raw (uint8 0/1),
      boundary_bwbl (uint8 0/1/2),
      stats dict
    """
    h = int(win.height)
    w = int(win.width)
    patch_transform = ds.window_transform(win)

    left, bottom, right, top = window_bounds(ds, win)
    bbox_geom = box(left, bottom, right, top)

    if sindex is None:
        geoms = gdf.geometry[gdf.geometry.intersects(bbox_geom)]
        if not include_holes:
            geoms = geoms.apply(drop_holes_geom)
        shapes = [(geom, 1) for geom in geoms.values if geom is not None and not geom.is_empty]
    else:
        cand_idx = list(sindex.intersection((left, bottom, right, top)))
        if cand_idx:
            geoms = gdf.geometry.iloc[cand_idx]
            geoms = geoms[geoms.intersects(bbox_geom)]
            if not include_holes:
                geoms = geoms.apply(drop_holes_geom)
            shapes = [(geom, 1) for geom in geoms.values if geom is not None and not geom.is_empty]
        else:
            shapes = []

    extent = rasterize(
        shapes=shapes,
        out_shape=(h, w),
        transform=patch_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    k = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(extent, cv2.MORPH_GRADIENT, k)
    boundary_raw = (grad > 0).astype(np.uint8)

    if ignore_enabled and ignore_apply_to_extent and ignore_radius_px > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ignore_radius_px + 1, 2 * ignore_radius_px + 1))
        ign = cv2.dilate(boundary_raw, k2, iterations=1)
        extent_ig = extent.copy().astype(np.uint8)
        extent_ig[ign > 0] = np.uint8(ignore_value)
    else:
        extent_ig = extent.copy().astype(np.uint8)

    if boundary_raw.any():
        sk = skeletonize(boundary_raw.astype(bool)).astype(np.uint8)
    else:
        sk = np.zeros_like(boundary_raw, dtype=np.uint8)

    bwbl = np.full_like(boundary_raw, np.uint8(bwbl_background_value), dtype=np.uint8)
    if bwbl_buffer_px > 0 and sk.any():
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bwbl_buffer_px + 1, 2 * bwbl_buffer_px + 1))
        buf = cv2.dilate(sk, k3, iterations=1)
        bwbl[buf > 0] = np.uint8(bwbl_buffer_value)
    bwbl[sk > 0] = np.uint8(bwbl_skeleton_value)

    border = max(0, int(pad_px))
    if border > 0:
        border = min(border, h // 2, w // 2)
        if border > 0:
            if ignore_enabled and ignore_apply_to_extent:
                iv = np.uint8(ignore_value)
                extent_ig[:border, :] = iv
                extent_ig[-border:, :] = iv
                extent_ig[:, :border] = iv
                extent_ig[:, -border:] = iv

            bv = np.uint8(bwbl_buffer_value)
            bwbl[:border, :] = bv
            bwbl[-border:, :] = bv
            bwbl[:, :border] = bv
            bwbl[:, -border:] = bv

    mask_ratio = float(extent.mean())
    edge_ratio = float((boundary_raw > 0).mean())
    skel_ratio = float((sk > 0).mean())
    ignore_ratio = float((extent_ig == np.uint8(ignore_value)).mean()) if ignore_enabled and ignore_apply_to_extent else 0.0

    stats = {
        "mask_ratio": mask_ratio,
        "edge_ratio": edge_ratio,
        "skeleton_ratio": skel_ratio,
        "ignore_ratio": ignore_ratio,
        "cand_geoms": float(len(shapes)),
    }
    return extent, extent_ig, boundary_raw, bwbl, stats


def apply_nodata_ignore_policy(
    extent_ig: np.ndarray,
    bwbl: np.ndarray,
    valid_u8: np.ndarray,
    extent_ignore_value: int,
    bwbl_ignore_value: int,
) -> Tuple[np.ndarray, np.ndarray]:
    invalid = valid_u8 == 0
    out_extent_ig = extent_ig.copy().astype(np.uint8)
    out_extent_ig[invalid] = np.uint8(extent_ignore_value)
    out_bwbl = bwbl.copy().astype(np.uint8)
    out_bwbl[invalid] = np.uint8(bwbl_ignore_value)
    return out_extent_ig, out_bwbl
