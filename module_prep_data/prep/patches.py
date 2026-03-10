from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import json
import numpy as np

import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box, Point, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, transform as shapely_transform
from pyproj import CRS, Transformer

import cv2
from skimage.morphology import skeletonize
from .utils import approx_utm_epsg_from_lonlat, unit_is_meter


# =========================
# Config
# =========================
@dataclass
class PatchConfig:
    patch_size_px: int
    train_crop_px: int
    pad_px: int

    center_weight: float
    boundary_weight: float

    negatives_ratio: float
    negatives_min_dist_m: float

    min_valid_ratio: float
    min_mask_ratio: float
    max_mask_ratio: float
    neg_max_mask_ratio: float  # for negative patches

    bwbl_buffer_px: int
    bwbl_background_value: int
    bwbl_skeleton_value: int
    bwbl_buffer_value: int

    build_extent: bool
    build_extent_ig: bool
    build_boundary_raw: bool
    build_boundary_bwbl: bool
    boundary_include_holes: bool

    ignore_radius_px: int
    ignore_enabled: bool
    ignore_value: int
    ignore_apply_to_extent: bool

    # ✅ NEW: NoData policy for valid_mask + valid_ratio
    nodata_value: float = 65536
    nodata_rule: str = "control-band"          # "control-band" | "all-bands"
    control_band_1based: int = 1               # only if rule == "control-band"

    # ✅ NEW: write valid mask
    write_valid_mask: bool = True

    # ✅ NEW: apply NoData ignore policy to targets
    apply_nodata_ignore_policy: bool = True    # valid=0 => extent_ig=255, bwbl=2
    nodata_ignore_extent_value: int = 255
    nodata_ignore_bwbl_value: int = 2

    seed: int = 123
    target_patches: int = 800


# =========================
# IO helpers
# =========================
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _pixel_size_m(ds: rasterio.DatasetReader) -> float:
    # average of |a| and |e| for north-up
    return float((abs(ds.transform.a) + abs(ds.transform.e)) / 2.0)


# =========================
# NoData policy
# =========================
def _valid_mask_from_chip(chip: np.ndarray, nodata_value: float, nodata_rule: str, control_band_1based: int) -> np.ndarray:
    """
    chip: (C,H,W) any numeric dtype
    returns uint8 mask (H,W): 1 valid, 0 nodata
    """
    rule = (nodata_rule or "control-band").strip().lower()
    nd = nodata_value

    if rule == "control-band":
        b = int(control_band_1based) - 1
        if b < 0 or b >= chip.shape[0]:
            raise RuntimeError(f"control_band_1based out of range: {control_band_1based} for chip with C={chip.shape[0]}")
        valid = (chip[b] != nd)
        return valid.astype(np.uint8)

    if rule == "all-bands":
        # nodata if all bands == nd
        all_nd = np.all(chip == nd, axis=0)
        valid = ~all_nd
        return valid.astype(np.uint8)

    raise RuntimeError(f"Unknown nodata_rule: {nodata_rule}")


def _valid_ratio_from_valid_mask(valid_u8: np.ndarray) -> float:
    return float(valid_u8.mean())


# =========================
# Negative distance checker
# =========================
def _build_negative_distance_checker(
    ds: rasterio.DatasetReader,
    union_geom,
    min_dist_m: float,
) -> Callable[[Point], bool]:
    """
    Returns callable(point_in_raster_crs) -> bool (is too close).
    For geographic raster CRS, distance is computed in auto-UTM (meters).
    """
    if union_geom is None or union_geom.is_empty or min_dist_m <= 0:
        return lambda _p: False

    try:
        raster_crs = CRS.from_user_input(ds.crs)
    except Exception:
        raster_crs = None

    # Fast path: projected CRS in meters.
    if raster_crs is not None and raster_crs.is_projected and unit_is_meter(raster_crs):
        too_close = union_geom.buffer(float(min_dist_m))
        return lambda p: bool(too_close.intersects(p))

    # Geographic or non-meter CRS -> convert to metric CRS (auto-UTM by centroid).
    if raster_crs is None:
        too_close = union_geom.buffer(float(min_dist_m))
        return lambda p: bool(too_close.intersects(p))

    try:
        to_wgs84 = Transformer.from_crs(raster_crs, CRS.from_epsg(4326), always_xy=True).transform
        centroid_4326 = shapely_transform(to_wgs84, union_geom.centroid)
        metric_epsg = approx_utm_epsg_from_lonlat(float(centroid_4326.x), float(centroid_4326.y))
        metric_crs = CRS.from_epsg(metric_epsg)

        to_metric = Transformer.from_crs(raster_crs, metric_crs, always_xy=True).transform
        union_metric = shapely_transform(to_metric, union_geom)
        too_close_metric = union_metric.buffer(float(min_dist_m))

        def _is_too_close(p: Point) -> bool:
            p_metric = shapely_transform(to_metric, p)
            return bool(too_close_metric.intersects(p_metric))

        return _is_too_close
    except Exception:
        too_close = union_geom.buffer(float(min_dist_m))
        return lambda p: bool(too_close.intersects(p))


# =========================
# Geometry helpers
# =========================
def _drop_holes_geom(geom):
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
            gg = _drop_holes_geom(g)
            if gg is not None and not gg.is_empty:
                out.append(gg)
        return GeometryCollection(out)

    return geom


def _safe_window_centered(col: int, row: int, w: int, h: int, width: int, height: int) -> Optional[Tuple[int, int]]:
    xoff = int(col - w // 2)
    yoff = int(row - h // 2)
    if xoff < 0 or yoff < 0:
        return None
    if xoff + w > width or yoff + h > height:
        return None
    return xoff, yoff


def _window_bounds(ds: rasterio.DatasetReader, win: Window) -> Tuple[float, float, float, float]:
    b = rasterio.windows.bounds(win, ds.transform)
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


# =========================
# Labels building
# =========================
def _extent_and_boundaries_for_window(
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
    H = int(win.height)
    W = int(win.width)
    patch_transform = ds.window_transform(win)

    left, bottom, right, top = _window_bounds(ds, win)
    bbox_geom = box(left, bottom, right, top)

    if sindex is None:
        geoms = gdf.geometry[gdf.geometry.intersects(bbox_geom)]
        if not include_holes:
            geoms = geoms.apply(_drop_holes_geom)
        shapes = [(geom, 1) for geom in geoms.values if geom is not None and not geom.is_empty]
    else:
        cand_idx = list(sindex.intersection((left, bottom, right, top)))
        if cand_idx:
            geoms = gdf.geometry.iloc[cand_idx]
            geoms = geoms[geoms.intersects(bbox_geom)]
            if not include_holes:
                geoms = geoms.apply(_drop_holes_geom)
            shapes = [(geom, 1) for geom in geoms.values if geom is not None and not geom.is_empty]
        else:
            shapes = []

    extent = rasterize(
        shapes=shapes,
        out_shape=(H, W),
        transform=patch_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    # boundary_raw via morphological gradient
    k = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(extent, cv2.MORPH_GRADIENT, k)
    boundary_raw = (grad > 0).astype(np.uint8)

    # extent_ig: ignore around boundary (0/1/ignore_value)
    if ignore_enabled and ignore_apply_to_extent and ignore_radius_px > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ignore_radius_px + 1, 2 * ignore_radius_px + 1))
        ign = cv2.dilate(boundary_raw, k2, iterations=1)
        extent_ig = extent.copy().astype(np.uint8)
        extent_ig[ign > 0] = np.uint8(ignore_value)
    else:
        extent_ig = extent.copy().astype(np.uint8)

    # BWBL: skeleton + buffer(ignore=2)
    if boundary_raw.any():
        sk = skeletonize(boundary_raw.astype(bool)).astype(np.uint8)
    else:
        sk = np.zeros_like(boundary_raw, dtype=np.uint8)

    bwbl = np.full_like(boundary_raw, np.uint8(bwbl_background_value), dtype=np.uint8)
    if bwbl_buffer_px > 0 and sk.any():
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bwbl_buffer_px + 1, 2 * bwbl_buffer_px + 1))
        buf = cv2.dilate(sk, k3, iterations=1)
        bwbl[buf > 0] = np.uint8(bwbl_buffer_value)
    bwbl[sk > 0] = np.uint8(bwbl_skeleton_value)  # skeleton wins

    # Optionally ignore the patch border band.
    border = max(0, int(pad_px))
    if border > 0:
        border = min(border, H // 2, W // 2)
        if border > 0:
            if ignore_enabled and ignore_apply_to_extent:
                iv = np.uint8(ignore_value)
                extent_ig[:border, :] = iv
                extent_ig[-border:, :] = iv
                extent_ig[:, :border] = iv
                extent_ig[:, -border:] = iv
            # BWBL uses buffer class as ignore class downstream.
            bv = np.uint8(bwbl_buffer_value)
            bwbl[:border, :] = bv
            bwbl[-border:, :] = bv
            bwbl[:, :border] = bv
            bwbl[:, -border:] = bv

    mask_ratio = float(extent.mean())  # extent is 0/1
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


# =========================
# TIFF writers
# =========================
def _write_geotiff_multiband(path: Path, chip: np.ndarray, ds: rasterio.DatasetReader, win: Window) -> None:
    _ensure_dir(path.parent)
    meta = ds.meta.copy()
    meta.update(
        {
            "height": int(win.height),
            "width": int(win.width),
            "transform": ds.window_transform(win),
            "count": int(chip.shape[0]),
        }
    )
    with rasterio.open(path, "w", **meta) as out:
        out.write(chip)


def _write_geotiff_mask(path: Path, arr2d: np.ndarray, ds: rasterio.DatasetReader, win: Window) -> None:
    _ensure_dir(path.parent)

    meta = {
        "driver": "GTiff",
        "height": int(win.height),
        "width": int(win.width),
        "count": 1,
        "dtype": str(arr2d.dtype),
        "crs": ds.crs,
        "transform": ds.window_transform(win),
        "compress": "DEFLATE",
        "tiled": True,
    }
    with rasterio.open(str(path), "w", **meta) as out:
        out.write(arr2d, 1)


# =========================
# Main
# =========================
def make_patches_for_dataset(
    raster_path: Path,
    vector_path: Path,
    out_root: Path,
    ds_name: str,
    cfg: PatchConfig,
    cleaned_vector_gpkg: Optional[Path] = None,
    vector_layer: Optional[str] = None,
) -> Path:
    """
    Writes patches into:
      out_root/<ds_name>/{img,extent,extent_ig,boundary_raw,boundary_bwbl,valid,meta}
    Returns path to manifest.json
    """
    rng = np.random.default_rng(cfg.seed)

    out_ds = out_root / ds_name
    img_dir = out_ds / "img"
    extent_dir = out_ds / "extent"
    extent_ig_dir = out_ds / "extent_ig"
    braw_dir = out_ds / "boundary_raw"
    bwbl_dir = out_ds / "boundary_bwbl"
    valid_dir = out_ds / "valid"
    meta_dir = out_ds / "meta"
    for d in [img_dir, extent_dir, extent_ig_dir, braw_dir, bwbl_dir, valid_dir, meta_dir]:
        _ensure_dir(d)

    # Load vector (support vector_layer)
    gdf = gpd.read_file(str(vector_path), layer=vector_layer) if vector_layer else gpd.read_file(str(vector_path))
    if gdf.empty:
        raise RuntimeError("Vector is empty")
    if gdf.crs is None:
        raise RuntimeError("Vector CRS missing")

    with rasterio.open(str(raster_path)) as ds:
        if ds.crs is None:
            raise RuntimeError("Raster CRS missing")
        raster_crs = ds.crs

        # Reproject vector to raster CRS (in-memory only)
        gdf = gdf.to_crs(raster_crs)
        gdf = gdf.explode(index_parts=False, ignore_index=True)
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if gdf.empty:
            raise RuntimeError("No polygon features left after CRS/geometry filtering")

        # Spatial index for fast bbox queries
        try:
            sindex = gdf.sindex
        except Exception:
            sindex = None

        union_geom = unary_union([g for g in gdf.geometry.values if g is not None and not g.is_empty])
        pix_m = _pixel_size_m(ds)

        target_total = int(cfg.target_patches)
        neg_ratio = float(cfg.negatives_ratio)
        pos_target = int(round(target_total * (1.0 - neg_ratio)))
        neg_target = int(target_total - pos_target)

        w_sum = max(1e-9, cfg.center_weight + cfg.boundary_weight)
        center_target = int(round(pos_target * (cfg.center_weight / w_sum)))
        boundary_target = int(pos_target - center_target)

        patch_w = patch_h = int(cfg.patch_size_px)
        if ds.width < patch_w or ds.height < patch_h:
            raise RuntimeError(f"Raster too small for patch_size={patch_w}: raster size is {ds.width}x{ds.height}")

        nfeat = len(gdf)
        if nfeat == 0:
            raise RuntimeError("No features after explode")

        center_feat_idx = rng.choice(nfeat, size=center_target, replace=(center_target > nfeat))
        boundary_feat_idx = rng.choice(nfeat, size=boundary_target, replace=(boundary_target > nfeat))

        def sample_point_in_poly(poly) -> Optional[Point]:
            minx, miny, maxx, maxy = poly.bounds
            for _ in range(200):
                x = float(rng.uniform(minx, maxx))
                y = float(rng.uniform(miny, maxy))
                p = Point(x, y)
                if poly.contains(p):
                    return p
            return None

        def sample_point_on_boundary(poly) -> Optional[Point]:
            b = poly.boundary
            if b is None or b.is_empty:
                return None
            try:
                length = float(b.length)
            except Exception:
                return None
            if length <= 0:
                return None
            t = float(rng.uniform(0.0, length))
            try:
                p = b.interpolate(t)
                if p is None or p.is_empty:
                    return None
                return p
            except Exception:
                return None

        def jitter_point(p: Point, jitter_m: float) -> Point:
            dx = float(rng.uniform(-jitter_m, jitter_m))
            dy = float(rng.uniform(-jitter_m, jitter_m))
            return Point(float(p.x + dx), float(p.y + dy))

        written = 0
        written_center = 0
        written_boundary = 0
        written_neg = 0

        manifest_rows: List[dict] = []
        rejects = {"oob": 0, "valid": 0, "mask": 0, "neg_dist": 0, "neg_mask": 0, "other": 0}

        def try_write_patch(
            patch_id: str,
            win: Window,
            inside_mode: str,
            feat_i: Optional[int],
        ) -> bool:
            nonlocal written, written_center, written_boundary, written_neg

            chip = ds.read(window=win)  # (C,H,W)

            # ✅ valid mask / valid ratio using POLICY (NOT ds.nodata)
            valid_u8 = _valid_mask_from_chip(
                chip,
                nodata_value=float(cfg.nodata_value),
                nodata_rule=str(cfg.nodata_rule),
                control_band_1based=int(cfg.control_band_1based),
            )
            vr = _valid_ratio_from_valid_mask(valid_u8)
            if vr < cfg.min_valid_ratio:
                rejects["valid"] += 1
                return False

            extent, extent_ig, braw, bwbl, st = _extent_and_boundaries_for_window(
                ds=ds,
                gdf=gdf,
                sindex=sindex,
                win=win,
                include_holes=cfg.boundary_include_holes,
                ignore_enabled=cfg.ignore_enabled,
                ignore_value=cfg.ignore_value,
                ignore_apply_to_extent=cfg.ignore_apply_to_extent,
                ignore_radius_px=cfg.ignore_radius_px,
                pad_px=cfg.pad_px,
                bwbl_buffer_px=cfg.bwbl_buffer_px,
                bwbl_background_value=cfg.bwbl_background_value,
                bwbl_skeleton_value=cfg.bwbl_skeleton_value,
                bwbl_buffer_value=cfg.bwbl_buffer_value,
            )
            mr = st["mask_ratio"]

            if inside_mode == "negative":
                if mr > cfg.neg_max_mask_ratio:
                    rejects["neg_mask"] += 1
                    return False
            else:
                if mr < cfg.min_mask_ratio or mr > cfg.max_mask_ratio:
                    rejects["mask"] += 1
                    return False

            # ✅ Apply NoData ignore policy on targets
            if cfg.apply_nodata_ignore_policy:
                invalid = (valid_u8 == 0)
                # extent_ig ignore value from config
                extent_ig = extent_ig.copy().astype(np.uint8)
                extent_ig[invalid] = np.uint8(cfg.nodata_ignore_extent_value)
                # boundary ignore value from config
                bwbl = bwbl.copy().astype(np.uint8)
                bwbl[invalid] = np.uint8(cfg.nodata_ignore_bwbl_value)

            # write files
            base = patch_id
            img_path = img_dir / f"img_{base}.tif"
            extent_path = extent_dir / f"extent_{base}.tif"
            extent_ig_path = extent_ig_dir / f"extent_ig_{base}.tif"
            braw_path = braw_dir / f"boundary_raw_{base}.tif"
            bwbl_path = bwbl_dir / f"bwbl_{base}.tif"
            valid_path = valid_dir / f"valid_{base}.tif"
            meta_path = meta_dir / f"meta_{base}.json"

            extent_out = extent.astype(np.uint8) if cfg.build_extent else np.zeros_like(extent, dtype=np.uint8)
            extent_ig_out = extent_ig.astype(np.uint8) if cfg.build_extent_ig else np.zeros_like(extent_ig, dtype=np.uint8)

            braw_out = braw.astype(np.uint8) if cfg.build_boundary_raw else np.zeros_like(braw, dtype=np.uint8)
            bwbl_out = bwbl.astype(np.uint8) if cfg.build_boundary_bwbl else np.full_like(
                bwbl, np.uint8(cfg.bwbl_background_value), dtype=np.uint8
            )

            _write_geotiff_multiband(img_path, chip, ds, win)
            _write_geotiff_mask(extent_path, extent_out, ds, win)
            _write_geotiff_mask(extent_ig_path, extent_ig_out, ds, win)
            _write_geotiff_mask(braw_path, braw_out, ds, win)
            _write_geotiff_mask(bwbl_path, bwbl_out, ds, win)

            if cfg.write_valid_mask:
                _write_geotiff_mask(valid_path, valid_u8.astype(np.uint8), ds, win)

            nodata_frac = float(1.0 - vr)

            meta = {
                "dataset": ds_name,
                "patch_id": base,
                "inside_mode": inside_mode,  # center | boundary | negative
                "feat_index": int(feat_i) if feat_i is not None else None,
                "xoff": int(win.col_off),
                "yoff": int(win.row_off),
                "w": int(win.width),
                "h": int(win.height),

                # ✅ critical: valid_ratio from policy, computed before any NoData remap
                "valid_ratio": float(vr),
                "nodata_frac": float(nodata_frac),

                "mask_ratio": float(mr),
                "edge_ratio": float(st["edge_ratio"]),
                "skeleton_ratio": float(st["skeleton_ratio"]),
                "ignore_ratio": float(st["ignore_ratio"]),
                "pixel_size_m": float(pix_m),
                "train_crop_px": int(cfg.train_crop_px),
                "pad_px": int(cfg.pad_px),

                "nodata_policy": {
                    "value": float(cfg.nodata_value),
                    "rule": str(cfg.nodata_rule),
                    "control_band_1based": int(cfg.control_band_1based),
                },

                "labels_written": {
                    "extent": bool(cfg.build_extent),
                    "extent_ig": bool(cfg.build_extent_ig),
                    "boundary_raw": bool(cfg.build_boundary_raw),
                    "boundary_bwbl": bool(cfg.build_boundary_bwbl),
                    "valid": bool(cfg.write_valid_mask),
                },
            }
            _write_json(meta_path, meta)

            manifest_rows.append(meta)

            written += 1
            if inside_mode == "center":
                written_center += 1
            elif inside_mode == "boundary":
                written_boundary += 1
            else:
                written_neg += 1
            return True

        # ---- Positive: center ----
        patch_idx = 0
        for fi in center_feat_idx:
            poly = gdf.geometry.iloc[int(fi)]
            p = sample_point_in_poly(poly)
            if p is None:
                rejects["other"] += 1
                continue
            row, col = ds.index(p.x, p.y)
            off = _safe_window_centered(col, row, patch_w, patch_h, ds.width, ds.height)
            if off is None:
                rejects["oob"] += 1
                continue
            xoff, yoff = off
            win = Window(xoff, yoff, patch_w, patch_h)
            pid = f"{ds_name}_{patch_idx:06d}"
            patch_idx += 1
            _ = try_write_patch(pid, win, "center", int(fi))

        # ---- Positive: boundary ----
        jitter_m = max(1.0, 0.10 * patch_w * pix_m)  # ~10% patch size
        for fi in boundary_feat_idx:
            poly = gdf.geometry.iloc[int(fi)]
            p0 = sample_point_on_boundary(poly)
            if p0 is None:
                rejects["other"] += 1
                continue
            p = jitter_point(p0, jitter_m=jitter_m)
            row, col = ds.index(p.x, p.y)
            off = _safe_window_centered(col, row, patch_w, patch_h, ds.width, ds.height)
            if off is None:
                rejects["oob"] += 1
                continue
            xoff, yoff = off
            win = Window(xoff, yoff, patch_w, patch_h)
            pid = f"{ds_name}_{patch_idx:06d}"
            patch_idx += 1
            _ = try_write_patch(pid, win, "boundary", int(fi))

        # ---- Negatives ----
        min_dist = float(cfg.negatives_min_dist_m)
        is_too_close = _build_negative_distance_checker(ds, union_geom, min_dist_m=min_dist)

        attempts = 0
        max_attempts = max(neg_target * 50, 5000)
        col_min = int(patch_w // 2)
        col_max = int(ds.width - (patch_w - patch_w // 2))
        row_min = int(patch_h // 2)
        row_max = int(ds.height - (patch_h - patch_h // 2))

        while written_neg < neg_target and attempts < max_attempts:
            attempts += 1
            col = int(rng.integers(col_min, col_max + 1))
            row = int(rng.integers(row_min, row_max + 1))

            x, y = ds.xy(row, col)
            p = Point(float(x), float(y))

            try:
                if is_too_close(p):
                    rejects["neg_dist"] += 1
                    continue
            except Exception:
                pass

            off = _safe_window_centered(col, row, patch_w, patch_h, ds.width, ds.height)
            if off is None:
                rejects["oob"] += 1
                continue
            xoff, yoff = off
            win = Window(xoff, yoff, patch_w, patch_h)

            pid = f"{ds_name}_{patch_idx:06d}"
            patch_idx += 1
            ok = try_write_patch(pid, win, "negative", None)
            if not ok:
                continue

        summary = {
            "dataset": ds_name,
            "raster": str(raster_path),
            "vector": str(vector_path),
            "vector_layer": vector_layer,
            "written_total": int(written),
            "written_center": int(written_center),
            "written_boundary": int(written_boundary),
            "written_negative": int(written_neg),
            "targets": {
                "target_total": int(target_total),
                "pos_target": int(pos_target),
                "center_target": int(center_target),
                "boundary_target": int(boundary_target),
                "neg_target": int(neg_target),
            },
            "rejects": rejects,
            "nodata_policy": {
                "value": float(cfg.nodata_value),
                "rule": str(cfg.nodata_rule),
                "control_band_1based": int(cfg.control_band_1based),
            },
        }

        manifest_path = out_ds / "manifest.json"
        _write_json(manifest_path, {"summary": summary, "patches": manifest_rows})

        if cleaned_vector_gpkg is not None:
            try:
                gdf.to_file(cleaned_vector_gpkg, driver="GPKG", layer="fields_raster_crs")
            except Exception:
                pass

        return manifest_path
