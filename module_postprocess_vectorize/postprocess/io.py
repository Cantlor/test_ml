from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.transform import Affine


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RasterMeta:
    """Metadata shared by co-registered rasters."""

    width: int
    height: int
    transform: Affine
    crs: CRS
    profile: Dict[str, Any]


@dataclass(frozen=True)
class InputRasters:
    """Loaded inputs for the post-processing pipeline."""

    extent_prob: np.ndarray
    boundary_prob: np.ndarray
    valid_mask: np.ndarray
    meta: RasterMeta
    valid_source: str
    valid_context: Dict[str, Any]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def write_yaml(path: Path, obj: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(obj), f, allow_unicode=True, sort_keys=False)


def deep_update(base: Dict[str, Any], patch: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively update nested mapping values."""
    out = dict(base)
    for key, value in patch.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_update(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _to_float_probability(arr: np.ndarray, name: str, target_dtype: np.dtype = np.float32) -> np.ndarray:
    """Validate or normalize probability raster to [0,1] with configurable float dtype."""
    out = arr.astype(np.float32, copy=False)
    finite = np.isfinite(out)
    if not np.all(finite):
        raise ValueError(f"{name}: raster contains NaN/Inf values")

    v_min = float(out.min())
    v_max = float(out.max())

    if v_min >= -1e-6 and v_max <= 1.0 + 1e-6:
        np.clip(out, 0.0, 1.0, out=out)
        return out.astype(target_dtype, copy=False)

    if v_min >= -1e-6 and v_max <= 255.0 + 1e-6:
        _LOG.warning("%s outside [0,1], scaling by 255 (min=%.5f max=%.5f)", name, v_min, v_max)
        out /= 255.0
        np.clip(out, 0.0, 1.0, out=out)
        return out.astype(target_dtype, copy=False)

    if v_min >= -1e-6 and v_max <= 100.0 + 1e-6:
        _LOG.warning("%s outside [0,1], scaling by 100 (min=%.5f max=%.5f)", name, v_min, v_max)
        out /= 100.0
        np.clip(out, 0.0, 1.0, out=out)
        return out.astype(target_dtype, copy=False)

    raise ValueError(
        f"{name}: probability range must be in [0,1] (or [0,255]/[0,100] for auto-scale), got min={v_min}, max={v_max}"
    )


def _read_single_band(path: Path, out_dtype: Optional[str] = None) -> tuple[np.ndarray, RasterMeta]:
    with rasterio.open(path) as ds:
        if ds.count < 1:
            raise ValueError(f"{path}: raster has no bands")
        if ds.crs is None:
            raise ValueError(f"{path}: CRS is missing")
        if out_dtype:
            arr = ds.read(1, out_dtype=out_dtype)
        else:
            arr = ds.read(1)
        profile = ds.profile.copy()
        meta = RasterMeta(
            width=int(ds.width),
            height=int(ds.height),
            transform=ds.transform,
            crs=ds.crs,
            profile=profile,
        )
    return arr, meta


def _load_valid_mask_from_footprint(
    path: Path,
    *,
    nodata_value: Optional[float],
    nodata_rule: str,
    control_band_1based: int,
) -> tuple[np.ndarray, RasterMeta, str, Dict[str, Any]]:
    with rasterio.open(path) as ds:
        if ds.count < 1:
            raise ValueError(f"{path}: raster has no bands")
        if ds.crs is None:
            raise ValueError(f"{path}: CRS is missing")

        effective_nodata = nodata_value
        if effective_nodata is None and ds.nodata is not None:
            effective_nodata = float(ds.nodata)

        if effective_nodata is None:
            mask_arr = ds.read(1, out_dtype="float32")
            valid_mask = (mask_arr > 0).astype(np.uint8)
            valid_source = "footprint_nonzero_fallback"
            resolved_rule = "nonzero"
        else:
            rule = str(nodata_rule or "control-band").strip().lower()
            if rule == "all-bands":
                valid_bool = np.ones((int(ds.height), int(ds.width)), dtype=bool)
                for band_idx in range(1, int(ds.count) + 1):
                    band_arr = ds.read(band_idx)
                    valid_bool &= band_arr != effective_nodata
            else:
                band_idx = max(1, min(int(ds.count), int(control_band_1based)))
                band_arr = ds.read(band_idx)
                valid_bool = band_arr != effective_nodata
                rule = "control-band"
                control_band_1based = band_idx
            valid_mask = valid_bool.astype(np.uint8)
            valid_source = "footprint_nodata"
            resolved_rule = rule

        profile = ds.profile.copy()
        meta = RasterMeta(
            width=int(ds.width),
            height=int(ds.height),
            transform=ds.transform,
            crs=ds.crs,
            profile=profile,
        )

    valid_context = {
        "path": str(path.resolve()),
        "nodata_value": effective_nodata,
        "nodata_rule": resolved_rule,
        "control_band_1based": int(control_band_1based),
    }
    return valid_mask, meta, valid_source, valid_context


def _assert_aligned(base: RasterMeta, other: RasterMeta, base_name: str, other_name: str) -> None:
    if base.width != other.width or base.height != other.height:
        raise ValueError(
            f"{other_name}: shape mismatch vs {base_name}: {(other.height, other.width)} != {(base.height, base.width)}"
        )
    if base.crs != other.crs:
        raise ValueError(f"{other_name}: CRS mismatch vs {base_name}: {other.crs} != {base.crs}")
    if not base.transform.almost_equals(other.transform, precision=1e-9):
        raise ValueError(f"{other_name}: affine transform mismatch vs {base_name}")


def load_inputs(
    extent_prob_path: Path,
    boundary_prob_path: Path,
    valid_mask_path: Optional[Path] = None,
    footprint_path: Optional[Path] = None,
    footprint_nodata_value: Optional[float] = None,
    footprint_nodata_rule: str = "control-band",
    footprint_control_band_1based: int = 1,
    prob_dtype: str = "float32",
) -> InputRasters:
    """
    Load and validate extent/boundary probability rasters (+ optional valid mask).

    If valid_mask_path is None, uses footprint_path as mask source if provided,
    otherwise all pixels are treated as valid.
    """
    extent_path = extent_prob_path.resolve()
    boundary_path = boundary_prob_path.resolve()

    np_prob_dtype = np.dtype(prob_dtype)
    if np_prob_dtype.kind != "f":
        raise ValueError(f"prob_dtype must be floating point, got {prob_dtype}")

    extent_raw, extent_meta = _read_single_band(extent_path, out_dtype=np_prob_dtype.name)
    boundary_raw, boundary_meta = _read_single_band(boundary_path, out_dtype=np_prob_dtype.name)
    _assert_aligned(extent_meta, boundary_meta, "extent_prob", "boundary_prob")

    extent_prob = _to_float_probability(extent_raw, "extent_prob", target_dtype=np_prob_dtype)
    boundary_prob = _to_float_probability(boundary_raw, "boundary_prob", target_dtype=np_prob_dtype)

    valid_mask: np.ndarray
    valid_source = "all_valid"
    valid_context: Dict[str, Any] = {}
    if valid_mask_path is not None:
        mask_arr, mask_meta = _read_single_band(Path(valid_mask_path).resolve(), out_dtype="uint8")
        _assert_aligned(extent_meta, mask_meta, "extent_prob", "valid_mask")
        valid_mask = (mask_arr > 0).astype(np.uint8)
        valid_source = "valid_mask"
        valid_context = {
            "path": str(Path(valid_mask_path).resolve()),
        }
    elif footprint_path is not None:
        valid_mask, mask_meta, valid_source, valid_context = _load_valid_mask_from_footprint(
            Path(footprint_path).resolve(),
            nodata_value=footprint_nodata_value,
            nodata_rule=footprint_nodata_rule,
            control_band_1based=footprint_control_band_1based,
        )
        _assert_aligned(extent_meta, mask_meta, "extent_prob", "footprint")
    else:
        valid_mask = np.ones_like(extent_prob, dtype=np.uint8)
        valid_context = {"mode": "all_valid"}

    # Hard background outside valid area.
    inv = valid_mask == 0
    extent_prob[inv] = np.array(0, dtype=extent_prob.dtype)
    boundary_prob[inv] = np.array(0, dtype=boundary_prob.dtype)

    return InputRasters(
        extent_prob=extent_prob,
        boundary_prob=boundary_prob,
        valid_mask=valid_mask,
        meta=extent_meta,
        valid_source=valid_source,
        valid_context=valid_context,
    )


def save_raster(
    path: Path,
    array: np.ndarray,
    meta: RasterMeta,
    dtype: str,
    nodata: Optional[float] = None,
    compress: str = "DEFLATE",
) -> Path:
    """Write a single-band GeoTIFF preserving CRS/transform."""
    out_path = path.resolve()
    ensure_dir(out_path.parent)

    profile = dict(meta.profile)
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        {
            "driver": "GTiff",
            "count": 1,
            "width": meta.width,
            "height": meta.height,
            "transform": meta.transform,
            "crs": meta.crs,
            "dtype": dtype,
            "compress": compress,
            "tiled": True,
            "bigtiff": "if_needed",
            "nodata": nodata,
        }
    )

    if bool(profile.get("tiled", False)):
        if meta.width < 16 or meta.height < 16:
            profile["tiled"] = False
        else:
            blockxsize = min(int(meta.width), 512)
            blockysize = min(int(meta.height), 512)
            blockxsize = max(16, (blockxsize // 16) * 16)
            blockysize = max(16, (blockysize // 16) * 16)
            profile["blockxsize"] = blockxsize
            profile["blockysize"] = blockysize

    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(array.astype(dtype, copy=False), 1)
    return out_path


def to_serializable(obj: Any) -> Any:
    """Convert nested structures to JSON-serializable form."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, Mapping):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    return obj
