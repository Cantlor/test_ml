#clip_raster.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from rasterio.mask import mask as rio_mask
from rasterio.transform import array_bounds
from rasterio.windows import Window, from_bounds
from shapely.ops import unary_union


@dataclass
class ClipResult:
    out_path: str
    mode: str
    bounds_used: Tuple[float, float, float, float]
    window: Tuple[int, int, int, int]  # col_off,row_off,width,height (в координатах исходного растра)
    wrote_mask_outside: bool


def _clamp_bounds_to_raster(bounds, ds) -> Tuple[float, float, float, float]:
    left, bottom, right, top = bounds
    rb = ds.bounds
    left = max(left, rb.left)
    bottom = max(bottom, rb.bottom)
    right = min(right, rb.right)
    top = min(top, rb.top)
    return (left, bottom, right, top)


def _normalize_bigtiff(v: str) -> str:
    v2 = (v or "").strip()
    m = {
        "yes": "YES",
        "no": "NO",
        "if_needed": "IF_NEEDED",
        "ifneeded": "IF_NEEDED",
        "IF_NEEDED": "IF_NEEDED",
        "YES": "YES",
        "NO": "NO",
    }
    if v2 in m:
        return m[v2]
    # если уже пришло что-то GDAL-совместимое — оставим как есть
    return v2


def _auto_utm_crs_from_lonlat(lon: float, lat: float) -> CRS:
    zone = int((lon + 180.0) // 6.0) + 1
    if lat >= 0:
        epsg = 32600 + zone  # WGS84 / UTM north
    else:
        epsg = 32700 + zone  # WGS84 / UTM south
    return CRS.from_epsg(epsg)


def _buffer_meters_safe(geom, raster_crs: CRS, buffer_m: float):
    if not buffer_m or buffer_m == 0.0:
        return geom

    if raster_crs.is_projected:
        return geom.buffer(float(buffer_m))

    # CRS географический (градусы) — буферим в метрах через авто-UTM
    c = geom.centroid
    lon, lat = float(c.x), float(c.y)
    utm = _auto_utm_crs_from_lonlat(lon, lat)

    fwd = Transformer.from_crs(raster_crs, utm, always_xy=True)
    inv = Transformer.from_crs(utm, raster_crs, always_xy=True)

    def _tx(transformer, g):
        return g.__class__(
            *(np.array(transformer.transform(*np.array(g.coords).T)).T)
        ) if hasattr(g, "coords") else g

    # Универсальнее: используем shapely ops через transform
    from shapely.ops import transform as shp_transform

    geom_utm = shp_transform(lambda x, y: fwd.transform(x, y), geom)
    geom_utm = geom_utm.buffer(float(buffer_m))
    geom_back = shp_transform(lambda x, y: inv.transform(x, y), geom_utm)
    return geom_back


def _clamp_window(win: Window, ds) -> Window:
    col_off = int(max(0, win.col_off))
    row_off = int(max(0, win.row_off))
    width = int(min(win.width, ds.width - col_off))
    height = int(min(win.height, ds.height - row_off))
    if width <= 0 or height <= 0:
        raise RuntimeError("Clip window is empty after clamping to raster bounds")
    return Window(col_off, row_off, width, height)


def clip_raster_by_vectors(
    raster_path: str,
    vector_path: str,
    out_path: str,
    mode: str = "bbox",          # "bbox" or "mask"
    buffer_m: float = 0.0,
    mask_outside: bool = False,  # only relevant when mode=="bbox"
    compress: str = "DEFLATE",
    tiled: bool = True,
    bigtiff: str = "if_needed",
    vector_layer: Optional[str] = None,
    nodata_value: Optional[float] = None,  # если ds.nodata None — используем это
) -> ClipResult:
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as ds:
        if ds.crs is None:
            raise RuntimeError("Raster CRS is missing")
        raster_crs = CRS.from_user_input(ds.crs)

        # nodata to use
        nd = ds.nodata if ds.nodata is not None else nodata_value

        gdf = gpd.read_file(vector_path, layer=vector_layer) if vector_layer else gpd.read_file(vector_path)
        if gdf.empty:
            raise RuntimeError("Vector is empty")
        if gdf.crs is None:
            raise RuntimeError("Vector CRS is missing")

        # Reproject vectors into raster CRS
        gdf = gdf.to_crs(raster_crs)

        # dissolve/union (holes сохраняются как часть геометрии)
        geom = unary_union([g for g in gdf.geometry.values if g is not None and not g.is_empty])
        if geom is None or geom.is_empty:
            raise RuntimeError("Union geometry is empty")

        # buffer in meters (safe)
        if buffer_m and buffer_m != 0.0:
            geom = _buffer_meters_safe(geom, raster_crs, float(buffer_m))

        bigtiff_norm = _normalize_bigtiff(bigtiff)

        if mode.lower() == "bbox":
            bounds = _clamp_bounds_to_raster(geom.bounds, ds)
            left, bottom, right, top = bounds

            win = from_bounds(left, bottom, right, top, transform=ds.transform)
            win = win.round_offsets().round_lengths()
            win = _clamp_window(win, ds)

            arr = ds.read(window=win)  # (C,H,W)
            transform = ds.window_transform(win)

            meta = ds.meta.copy()
            meta.update(
                {
                    "height": int(arr.shape[1]),
                    "width": int(arr.shape[2]),
                    "transform": transform,
                    "compress": compress,
                    "tiled": bool(tiled),
                }
            )
            if nd is not None:
                meta["nodata"] = nd
            if bigtiff_norm:
                meta["BIGTIFF"] = bigtiff_norm

            wrote_mask = False

            if mask_outside:
                if nd is None:
                    raise RuntimeError("mask_outside=True requires nodata (ds.nodata or nodata_value)")
                # Растеризуем маску В ТОЙ ЖЕ СЕТКЕ окна (bbox) и зануляем/ставим nodata снаружи
                h, w = int(win.height), int(win.width)
                inside = rasterize(
                    [(geom, 1)],
                    out_shape=(h, w),
                    transform=transform,
                    fill=0,
                    dtype="uint8",
                    all_touched=False,
                ).astype(bool)

                # outside -> nodata
                outside = ~inside
                # arr: (C,H,W)
                arr = arr.copy()
                # корректно по dtype
                fill_val = np.array(nd, dtype=arr.dtype)
                arr[:, outside] = fill_val
                wrote_mask = True

            with rasterio.open(out_path, "w", **meta) as out:
                out.write(arr)

            out_bounds = array_bounds(int(arr.shape[1]), int(arr.shape[2]), transform)
            return ClipResult(
                out_path=str(out_path_p),
                mode="bbox_mask" if wrote_mask else "bbox",
                bounds_used=(float(out_bounds[0]), float(out_bounds[1]), float(out_bounds[2]), float(out_bounds[3])),
                window=(int(win.col_off), int(win.row_off), int(win.width), int(win.height)),
                wrote_mask_outside=wrote_mask,
            )

        elif mode.lower() == "mask":
            if nd is None:
                raise RuntimeError("mode='mask' requires nodata (ds.nodata or nodata_value)")

            masked, m_transform = rio_mask(
                ds,
                shapes=[geom],
                crop=True,        # tight bounds around geom
                filled=True,
                nodata=nd,
                all_touched=False,
            )

            meta = ds.meta.copy()
            meta.update(
                {
                    "height": int(masked.shape[1]),
                    "width": int(masked.shape[2]),
                    "transform": m_transform,
                    "compress": compress,
                    "tiled": bool(tiled),
                    "nodata": nd,
                }
            )
            if bigtiff_norm:
                meta["BIGTIFF"] = bigtiff_norm

            with rasterio.open(out_path, "w", **meta) as out:
                out.write(masked)

            out_bounds = array_bounds(int(masked.shape[1]), int(masked.shape[2]), m_transform)

            # приблизительная “ссылка” на исходный растр через bbox bounds -> window
            win2 = from_bounds(out_bounds[0], out_bounds[1], out_bounds[2], out_bounds[3], transform=ds.transform)
            win2 = win2.round_offsets().round_lengths()
            win2 = _clamp_window(win2, ds)

            return ClipResult(
                out_path=str(out_path_p),
                mode="mask",
                bounds_used=(float(out_bounds[0]), float(out_bounds[1]), float(out_bounds[2]), float(out_bounds[3])),
                window=(int(win2.col_off), int(win2.row_off), int(win2.width), int(win2.height)),
                wrote_mask_outside=True,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'bbox' or 'mask'.")