from __future__ import annotations

from typing import Optional

import geopandas as gpd
import numpy as np
from pyproj import CRS as PyCRS
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

try:
    from shapely import make_valid as _make_valid_fn
except ImportError:  # pragma: no cover
    _make_valid_fn = None


def _make_valid(geom):
    if geom is None:
        return geom
    if _make_valid_fn is not None:
        return _make_valid_fn(geom)
    return geom.buffer(0)


def _strip_holes(geom):
    if geom is None or geom.is_empty:
        return geom
    if isinstance(geom, Polygon):
        return Polygon(geom.exterior)
    if isinstance(geom, MultiPolygon):
        polys = [Polygon(p.exterior) for p in geom.geoms if not p.is_empty and p.area > 0]
        if not polys:
            return GeometryCollection()
        return MultiPolygon(polys)
    if isinstance(geom, GeometryCollection):
        polys = [_strip_holes(g) for g in geom.geoms]
        polys = [g for g in polys if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        if not polys:
            return GeometryCollection()
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon([p for g in polys for p in (g.geoms if isinstance(g, MultiPolygon) else [g])])
    return geom


def _is_metric_crs(crs) -> bool:
    pycrs = PyCRS.from_user_input(crs)
    if not pycrs.is_projected:
        return False
    axis = pycrs.axis_info
    if not axis:
        return False
    unit = (axis[0].unit_name or "").lower()
    return ("metre" in unit) or ("meter" in unit) or (unit == "m")


def _estimate_local_metric_crs(gdf: gpd.GeoDataFrame):
    if gdf.crs is None:
        raise ValueError("GeoDataFrame CRS is required")

    if _is_metric_crs(gdf.crs):
        return gdf.crs

    ll = gdf.to_crs(4326)
    centroid = ll.unary_union.centroid
    lon = float(centroid.x)
    lat = float(centroid.y)

    zone = int(np.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def _dominant_angle_deg(geom) -> float:
    try:
        mrr = geom.minimum_rotated_rectangle
        coords = list(mrr.exterior.coords)
        if len(coords) < 2:
            return 0.0
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        return float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    except Exception:
        return 0.0


def _straighten_geometry(geom, snap_angle_deg: float):
    if geom is None or geom.is_empty:
        return geom
    step = float(snap_angle_deg)
    if step <= 0:
        return geom

    angle = _dominant_angle_deg(geom)
    snapped = round(angle / step) * step
    delta = snapped - angle
    if abs(delta) < 1e-6:
        return geom
    return affinity.rotate(geom, delta, origin="centroid", use_radians=False)


def clean_polygons(
    raw_gdf: gpd.GeoDataFrame,
    min_area_m2: float,
    simplify_m: float,
    remove_holes: bool,
    clip_geom=None,
    straighten_cfg: Optional[dict] = None,
) -> gpd.GeoDataFrame:
    """Run final geometry cleaning and return GIS-friendly polygons."""
    if raw_gdf.empty:
        out = raw_gdf.copy()
        out["area_m2"] = []
        out["field_id"] = []
        return out

    gdf = raw_gdf.copy()
    gdf = gdf[~gdf.geometry.isna()].copy()
    gdf["geometry"] = gdf.geometry.apply(_make_valid)
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if clip_geom is not None:
        gdf["geometry"] = gdf.geometry.intersection(clip_geom)
        gdf = gdf[~gdf.geometry.is_empty].copy()

    if remove_holes:
        gdf["geometry"] = gdf.geometry.apply(_strip_holes)
        gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        gdf["area_m2"] = []
        gdf["field_id"] = []
        return gdf

    metric_crs = _estimate_local_metric_crs(gdf)
    gdf_m = gdf.to_crs(metric_crs)

    if float(simplify_m) > 0:
        gdf_m["geometry"] = gdf_m.geometry.simplify(float(simplify_m), preserve_topology=True)

    straighten_cfg = straighten_cfg or {}
    if bool(straighten_cfg.get("enabled", False)):
        snap_angle = float(straighten_cfg.get("snap_angle_deg", 15.0))
        gdf_m["geometry"] = gdf_m.geometry.apply(lambda g: _straighten_geometry(g, snap_angle_deg=snap_angle))

    gdf_m["geometry"] = gdf_m.geometry.apply(_make_valid)
    gdf_m = gdf_m[~gdf_m.geometry.is_empty].copy()

    gdf_m["area_m2"] = gdf_m.area.astype(float)
    if float(min_area_m2) > 0:
        gdf_m = gdf_m[gdf_m["area_m2"] >= float(min_area_m2)].copy()

    if gdf_m.empty:
        out = gdf_m.to_crs(raw_gdf.crs)
        out["field_id"] = []
        return out

    gdf_out = gdf_m.to_crs(raw_gdf.crs)
    gdf_out["area_m2"] = gdf_m["area_m2"].to_numpy(dtype=float)
    gdf_out = gdf_out.reset_index(drop=True)
    gdf_out["field_id"] = np.arange(1, len(gdf_out) + 1, dtype=np.int32)
    return gdf_out


def count_holes(gdf: gpd.GeoDataFrame) -> int:
    def _geom_holes(geom) -> int:
        if geom is None or geom.is_empty:
            return 0
        if isinstance(geom, Polygon):
            return len(geom.interiors)
        if isinstance(geom, MultiPolygon):
            return int(sum(len(p.interiors) for p in geom.geoms))
        return 0

    if gdf.empty:
        return 0
    return int(sum(_geom_holes(g) for g in gdf.geometry))
