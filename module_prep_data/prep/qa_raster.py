from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window


@dataclass
class RasterInfo:
    path: str
    crs: Optional[str]
    is_projected: Optional[bool]
    width: int
    height: int
    count: int
    dtypes: list
    nodata: Optional[float]
    transform: tuple
    bounds: tuple


def read_raster_info(path: str) -> RasterInfo:
    with rasterio.open(path) as ds:
        crs = ds.crs.to_string() if ds.crs else None
        is_projected = ds.crs.is_projected if ds.crs else None
        return RasterInfo(
            path=path,
            crs=crs,
            is_projected=is_projected,
            width=ds.width,
            height=ds.height,
            count=ds.count,
            dtypes=list(ds.dtypes),
            nodata=ds.nodata,
            transform=tuple(ds.transform),
            bounds=tuple(ds.bounds),
        )


def estimate_valid_ratio(
    path: str,
    nodata_value: Optional[float],
    nodata_rule: str = "control-band",   # "control-band" | "all-bands"
    control_band_1based: int = 1,
    sample_target_pixels: int = 2_000_000,
    window_size: int = 512,
    seed: int = 123,
) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    Оценка valid_ratio по spec-правилу NoData.

    valid pixel:
      - rule="control-band": valid = (band[control] != nodata_value)
      - rule="all-bands":    valid = NOT(all bands == nodata_value)

    sampling-based (рандомные окна), чтобы не читать весь растр.
    """
    if nodata_value is None:
        return None, {"skipped": True, "reason": "nodata_value_is_none"}

    rule = (nodata_rule or "control-band").strip().lower()
    if rule not in ("control-band", "all-bands"):
        return None, {"skipped": True, "reason": f"unknown_nodata_rule:{nodata_rule}"}

    rng = np.random.default_rng(seed)

    with rasterio.open(path) as ds:
        if ds.width < 2 or ds.height < 2:
            return None, {"skipped": True, "reason": "raster_too_small"}

        ws = int(min(window_size, ds.width, ds.height))
        max_x = int(ds.width - ws)
        max_y = int(ds.height - ws)

        total_px = 0
        valid_px = 0
        nodata_px = 0
        n_windows = 0

        # индексация контрольного бэнда
        cb = int(control_band_1based) - 1
        if rule == "control-band" and not (0 <= cb < ds.count):
            return None, {"skipped": True, "reason": f"control_band_out_of_range:{control_band_1based}", "count": ds.count}

        nd = nodata_value

        # читаем окна без masked=True — чтобы не зависеть от ds.nodata и GDAL-метаданных
        while total_px < sample_target_pixels and n_windows < 2000:
            xoff = int(rng.integers(0, max(1, max_x + 1)))
            yoff = int(rng.integers(0, max(1, max_y + 1)))
            win = Window(xoff, yoff, ws, ws)

            # (C,H,W)
            arr = ds.read(window=win)

            if rule == "control-band":
                vmask = (arr[cb] != nd)
            else:
                # nodata if ALL bands == nd
                vmask = ~np.all(arr == nd, axis=0)

            v = int(vmask.sum())
            px = int(vmask.size)
            ndc = px - v

            valid_px += v
            nodata_px += ndc
            total_px += px
            n_windows += 1

        ratio = (valid_px / total_px) if total_px > 0 else None
        meta = {
            "skipped": False,
            "nodata_value_used": float(nodata_value),
            "nodata_rule": nodata_rule,
            "control_band_1based": int(control_band_1based),
            "sample_target_pixels": int(sample_target_pixels),
            "window_size": int(ws),
            "windows_used": int(n_windows),
            "sampled_pixels": int(total_px),
            "valid_pixels": int(valid_px),
            "nodata_pixels": int(nodata_px),
            "nodata_frac": float(nodata_px / total_px) if total_px > 0 else None,
        }
        return ratio, meta