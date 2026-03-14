from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import rasterio
from rasterio.windows import Window


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def pixel_size_m(ds: rasterio.DatasetReader) -> float:
    return float((abs(ds.transform.a) + abs(ds.transform.e)) / 2.0)


def write_geotiff_multiband(path: Path, chip: np.ndarray, ds: rasterio.DatasetReader, win: Window) -> None:
    ensure_dir(path.parent)
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


def write_geotiff_mask(path: Path, arr2d: np.ndarray, ds: rasterio.DatasetReader, win: Window) -> None:
    ensure_dir(path.parent)
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
