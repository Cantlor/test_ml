from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Literal

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import CRS

from .utils import approx_utm_epsg_from_lonlat, unit_is_meter

FixInvalidMode = Literal["none", "make_valid_in_memory", "buffer0_in_memory"]


def _make_valid_series_in_memory(gdf: gpd.GeoDataFrame, mode: FixInvalidMode) -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    IMPORTANT:
      - НИЧЕГО не пишем на диск.
      - Это подготовка "in-memory" для растеризации/геом.операций.
    """
    info: Dict[str, Any] = {"mode": mode}

    invalid_before = int((~gdf.geometry.is_valid).sum())
    info["invalid_before"] = invalid_before

    if mode == "none" or invalid_before == 0:
        info["method"] = "none"
        info["invalid_after"] = invalid_before
        return gdf, info

    mv = None
    if mode == "make_valid_in_memory":
        try:
            from shapely import make_valid  # shapely>=2
            mv = make_valid
            info["method"] = "shapely.make_valid"
        except Exception:
            # fallback
            mode = "buffer0_in_memory"
            info["mode"] = mode

    if mode == "buffer0_in_memory":
        info["method"] = "buffer(0)"

    def fix_geom(geom):
        if geom is None or geom.is_empty:
            return geom
        try:
            if mv is not None:
                return mv(geom)
            return geom.buffer(0)
        except Exception:
            return geom

    gdf2 = gdf.copy()
    gdf2["geometry"] = gdf2.geometry.apply(fix_geom)

    invalid_after = int((~gdf2.geometry.is_valid).sum())
    info["invalid_after"] = invalid_after
    return gdf2, info


def _choose_metric_crs(gdf_in_raster_crs: gpd.GeoDataFrame, raster_crs: CRS) -> CRS:
    # If raster CRS is projected and unit is meter => ok
    if raster_crs is not None and raster_crs.is_projected and unit_is_meter(raster_crs):
        return raster_crs

    # Otherwise pick auto UTM based on centroid in WGS84
    g4326 = gdf_in_raster_crs
    if raster_crs is not None and raster_crs.to_epsg() != 4326:
        try:
            g4326 = gdf_in_raster_crs.to_crs(epsg=4326)
        except Exception:
            g4326 = gdf_in_raster_crs

    c = g4326.geometry.unary_union.centroid
    lon, lat = float(c.x), float(c.y)
    epsg = approx_utm_epsg_from_lonlat(lon, lat)
    return CRS.from_epsg(epsg)


@dataclass
class VectorInfo:
    path: str
    crs: Optional[str]
    features_raw: int
    features_after: int

    # QA stats
    invalid_before: int
    invalid_after: int
    empty_dropped: int
    nonpoly_dropped: int
    dropped_small: int
    clipped_out: int

    min_area_m2: float
    raster_crs_used: str
    metric_crs_used: str

    # notes
    geometry_was_clipped: bool
    gt_modified_on_disk: bool


def check_and_prepare_vector(
    vector_path: str,
    raster_bounds: tuple,   # (left, bottom, right, top)
    raster_crs_str: str,
    min_area_m2: float,
    vector_layer: Optional[str] = None,

    # IMPORTANT: по умолчанию GT не “чиним”
    fix_invalid_mode: FixInvalidMode = "none",

    drop_empty: bool = True,
    explode_multipolygons: bool = True,

    # IMPORTANT: по умолчанию НЕ делаем intersection(), только фильтруем intersects(bbox)
    clip_to_bounds: bool = True,
    clip_geometry: bool = False,

    keep_holes: bool = True,  # holes сохраняются автоматически, ничего делать не нужно
) -> Tuple[gpd.GeoDataFrame, VectorInfo, Dict[str, Any]]:
    extra: Dict[str, Any] = {"gt_modified_on_disk": False, "keep_holes": bool(keep_holes)}

    gdf = gpd.read_file(vector_path, layer=vector_layer) if vector_layer else gpd.read_file(vector_path)
    features_raw = int(len(gdf))
    if gdf.empty:
        raise RuntimeError("Vector is empty")

    if gdf.crs is None:
        raise RuntimeError("Vector CRS is None (missing .prj or metadata)")

    src_crs_str = gdf.crs.to_string()

    raster_crs = CRS.from_user_input(raster_crs_str)

    # Reproject vector into raster CRS for alignment
    gdf = gdf.to_crs(raster_crs)

    # Drop empty geometries
    empty_dropped = 0
    if drop_empty:
        before = len(gdf)
        gdf = gdf[gdf.geometry.notna()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()
        empty_dropped = int(before - len(gdf))
        if gdf.empty:
            raise RuntimeError("Vector has no non-empty geometries after drop_empty")

    # Keep only polygons (Polygon/MultiPolygon)
    before = len(gdf)
    gtypes = gdf.geometry.geom_type
    mask_poly = gtypes.isin(["Polygon", "MultiPolygon"])
    gdf = gdf[mask_poly].copy()
    nonpoly_dropped = int(before - len(gdf))
    if gdf.empty:
        raise RuntimeError("Vector has no Polygon/MultiPolygon geometries after type filter")

    # Explode multipolygons (как “подготовка датасета”, не перезаписываем GT)
    if explode_multipolygons:
        # сохраним связь с исходной записью
        gdf = gdf.reset_index(drop=False).rename(columns={"index": "orig_fid"})
        gdf = gdf.explode(index_parts=True, ignore_index=True)
        # index_parts=True добавляет multi-index; после explode он уже развёрнут, но orig_fid остаётся

    # Invalid stats (до фикса)
    invalid_before = int((~gdf.geometry.is_valid).sum())

    # In-memory fix (если явно включено)
    gdf_fixed, fix_info = _make_valid_series_in_memory(gdf, fix_invalid_mode)
    invalid_after = int((~gdf_fixed.geometry.is_valid).sum())
    extra["fix_invalid"] = fix_info

    # Clip/filter to raster bounds
    clipped_out = 0
    geometry_was_clipped = False
    if clip_to_bounds:
        left, bottom, right, top = raster_bounds
        bbox = box(left, bottom, right, top)

        before = len(gdf_fixed)
        gdf_fixed = gdf_fixed[gdf_fixed.geometry.intersects(bbox)].copy()
        clipped_out = int(before - len(gdf_fixed))
        if gdf_fixed.empty:
            raise RuntimeError("No vector features left after clip_to_raster_bounds filter")

        # ВАЖНО: intersection меняет геометрию — по умолчанию выключено.
        if clip_geometry:
            try:
                gdf_fixed["geometry"] = gdf_fixed.geometry.intersection(bbox)
                gdf_fixed = gdf_fixed[~gdf_fixed.geometry.is_empty].copy()
                geometry_was_clipped = True
            except Exception as e:
                extra["clip_geometry_error"] = str(e)

    # Area filter in metric CRS
    metric_crs = _choose_metric_crs(gdf_fixed, raster_crs)
    gdf_metric = gdf_fixed.to_crs(metric_crs)

    areas = gdf_metric.geometry.area
    mask_keep = areas >= float(min_area_m2)
    dropped_small = int((~mask_keep).sum())
    gdf_fixed = gdf_fixed.loc[mask_keep.values].copy()
    if gdf_fixed.empty:
        raise RuntimeError(f"No vector features left after min_area_m2 filter ({min_area_m2})")

    info = VectorInfo(
        path=vector_path,
        crs=src_crs_str,
        features_raw=features_raw,
        features_after=int(len(gdf_fixed)),

        invalid_before=invalid_before,
        invalid_after=invalid_after,
        empty_dropped=empty_dropped,
        nonpoly_dropped=nonpoly_dropped,
        dropped_small=dropped_small,
        clipped_out=clipped_out,

        min_area_m2=float(min_area_m2),
        raster_crs_used=raster_crs.to_string(),
        metric_crs_used=metric_crs.to_string(),

        geometry_was_clipped=bool(geometry_was_clipped),
        gt_modified_on_disk=False,
    )

    extra["geometry_was_clipped"] = bool(geometry_was_clipped)
    extra["note"] = "GT is never overwritten; any fixes/clips are in-memory only."

    return gdf_fixed, info, extra