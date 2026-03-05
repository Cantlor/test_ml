from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import math


def abs_from(base: Path, p: str | Path) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


def find_single_by_globs(root: Path, globs: List[str], require_single: bool = True) -> Tuple[Optional[Path], List[Path]]:
    """
    Возвращает (single_match_or_None, all_matches_sorted_unique)
    """
    matches: List[Path] = []
    for g in globs:
        matches.extend(root.glob(g))

    # unique + files only
    uniq = []
    seen = set()
    for p in matches:
        if not p.is_file():
            continue
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)

    uniq = sorted(uniq)

    if require_single:
        if len(uniq) != 1:
            return None, uniq
        return uniq[0], uniq
    else:
        return (uniq[0] if uniq else None), uniq


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def approx_utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    """
    WGS84 UTM zone, EPSG:326xx/327xx
    """
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    zone = max(1, min(60, zone))
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone


def unit_is_meter(pyproj_crs) -> bool:
    """
    Проверяем, что единицы CRS — метры.
    Работает и для pyproj.CRS и для rasterio CRS-объектов (если у них есть axis_info).
    """
    try:
        axis_info = getattr(pyproj_crs, "axis_info", None)
        if axis_info and len(axis_info) > 0:
            u = (axis_info[0].unit_name or "").lower()
            return ("metre" in u) or ("meter" in u) or (u == "m")
    except Exception:
        pass
    return False


def safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def safe_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default