from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import Point
from shapely.ops import transform as shapely_transform

from ..utils import approx_utm_epsg_from_lonlat, unit_is_meter


def build_negative_distance_checker(
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

    if raster_crs is not None and raster_crs.is_projected and unit_is_meter(raster_crs):
        too_close = union_geom.buffer(float(min_dist_m))
        return lambda p: bool(too_close.intersects(p))

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


def build_field_ids(
    ds_name: str,
    gdf: gpd.GeoDataFrame,
    vector_id_field: Optional[str],
) -> Tuple[List[str], str]:
    nfeat = len(gdf)
    field_id_source = "generated_from_feat_index"
    field_ids: List[str] = [f"{ds_name}::{i}" for i in range(nfeat)]

    if vector_id_field and vector_id_field in gdf.columns:
        field_id_source = f"vector_id_field:{vector_id_field}"
        raw_vals = gdf[vector_id_field]
        for i in range(nfeat):
            v = raw_vals.iloc[i]
            s = str(v).strip() if v is not None else ""
            if s and s.lower() not in {"none", "nan"}:
                field_ids[i] = f"{ds_name}::{s}"
    elif "orig_fid" in gdf.columns:
        field_id_source = "orig_fid"
        raw_vals = gdf["orig_fid"]
        for i in range(nfeat):
            v = raw_vals.iloc[i]
            s = str(v).strip() if v is not None else ""
            if s and s.lower() not in {"none", "nan"}:
                field_ids[i] = f"{ds_name}::{s}"

    return field_ids, field_id_source


def sample_point_in_poly(poly, rng: np.random.Generator, max_attempts: int = 200) -> Optional[Point]:
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_attempts):
        x = float(rng.uniform(minx, maxx))
        y = float(rng.uniform(miny, maxy))
        p = Point(x, y)
        if poly.contains(p):
            return p
    return None


def sample_point_on_boundary(poly, rng: np.random.Generator) -> Optional[Point]:
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


def jitter_point(p: Point, rng: np.random.Generator, jitter_m: float) -> Point:
    dx = float(rng.uniform(-jitter_m, jitter_m))
    dy = float(rng.uniform(-jitter_m, jitter_m))
    return Point(float(p.x + dx), float(p.y + dy))
