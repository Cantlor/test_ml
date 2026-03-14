from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union


def labels_to_geodataframe(labels: np.ndarray, transform: Affine, crs: CRS) -> gpd.GeoDataFrame:
    """Vectorize integer label raster into polygons (one row per label component)."""
    arr = labels.astype(np.int32, copy=False)
    mask = arr > 0

    records = []
    for geom_json, value in shapes(arr, mask=mask, transform=transform):
        label_id = int(value)
        if label_id <= 0:
            continue
        geom = shape(geom_json)
        if geom.is_empty:
            continue
        records.append({"label_id": label_id, "geometry": geom})

    if not records:
        return gpd.GeoDataFrame({"label_id": [], "geometry": []}, geometry="geometry", crs=crs)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)

    # Keep one object per label after cleanup.
    gdf = gdf.dissolve(by="label_id", as_index=False)
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    return gdf


def valid_mask_to_geometry(valid_mask: np.ndarray, transform: Affine) -> Optional[Polygon | MultiPolygon]:
    """Convert valid mask to union geometry for optional clipping."""
    mask = valid_mask.astype(np.uint8)
    if int(mask.sum()) == 0:
        return None

    geoms = []
    for geom_json, value in shapes(mask, mask=mask.astype(bool), transform=transform):
        if int(value) != 1:
            continue
        geom = shape(geom_json)
        if not geom.is_empty:
            geoms.append(geom)

    if not geoms:
        return None
    return unary_union(geoms)


def clip_geodataframe_to_geom(gdf: gpd.GeoDataFrame, clip_geom) -> gpd.GeoDataFrame:
    if gdf.empty or clip_geom is None:
        return gdf
    out = gdf.copy()
    out["geometry"] = out.geometry.intersection(clip_geom)
    out = out[~out.geometry.is_empty].copy()
    out = out.reset_index(drop=True)
    return out


def save_geodataframe(gdf: gpd.GeoDataFrame, path: Path) -> Path:
    out = path.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    ext = out.suffix.lower()
    if ext == ".gpkg":
        gdf.to_file(out, driver="GPKG")
    elif ext == ".shp":
        gdf.to_file(out, driver="ESRI Shapefile")
    else:
        gdf.to_file(out)
    return out
