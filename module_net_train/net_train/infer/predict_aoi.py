from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
from rasterio.windows import Window

import torch

from net_train.data.stats import NormalizationStats, normalize_image
from net_train.hardware import RuntimePlan, amp_dtype_from_plan
from net_train.infer.tiling import TileWindow, blend_weights, generate_windows
from net_train.utils.io import read_json


def _resolve_from_manifest_obj(obj: object, dataset_key: str) -> Path | None:
    # Preferred strict manifest from module_prep_data:
    # {"datasets": [{"dataset": "...", "status": "clipped", "aoi_raster_path": "..."}]}
    if isinstance(obj, dict) and isinstance(obj.get("datasets"), list):
        for it in obj["datasets"]:
            if not isinstance(it, dict):
                continue
            if str(it.get("dataset")) != dataset_key:
                continue
            status = str(it.get("status", "")).strip().lower()
            aoi_path = it.get("aoi_raster_path")
            if status and status != "clipped":
                continue
            if isinstance(aoi_path, str) and aoi_path.strip():
                candidate = Path(aoi_path).resolve()
                if candidate.exists():
                    return candidate

    # Preferred contract: {"dataset_key": "/abs/path/to/aoi.tif"}
    if isinstance(obj, dict) and dataset_key in obj and isinstance(obj[dataset_key], str):
        candidate = Path(obj[dataset_key]).resolve()
        if candidate.exists():
            return candidate

    # Backward-compatible fallback for old manifest format {"results": [...]}.
    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
        for it in obj["results"]:
            if isinstance(it, dict) and str(it.get("dataset")) == dataset_key:
                out = it.get("out_raster")
                if out:
                    candidate = Path(out).resolve()
                    if candidate.exists():
                        return candidate
    return None


def resolve_aoi_path(manifest_path: Path, dataset_key: str) -> Path:
    manifest_path = Path(manifest_path).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"AOI manifest not found: {manifest_path}")

    obj = read_json(manifest_path)
    resolved = _resolve_from_manifest_obj(obj, dataset_key)
    if resolved is not None:
        return resolved

    # Backward-compatible fallbacks for older layouts.
    checked = [manifest_path]
    alt_candidates = [
        manifest_path.parent / "aoi_rasters_manifest.json",
        manifest_path.parent / "aoi_rasters" / "aoi_rasters_manifest.json",
    ]
    for alt in alt_candidates:
        if alt in checked or not alt.exists():
            continue
        checked.append(alt)
        alt_obj = read_json(alt)
        resolved = _resolve_from_manifest_obj(alt_obj, dataset_key)
        if resolved is not None:
            return resolved

    raise KeyError(
        f"Cannot resolve existing AOI path for dataset_key='{dataset_key}'. "
        f"Checked manifests: {', '.join(str(p) for p in checked)}"
    )


def _batch_iter(seq: List[TileWindow], batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i:i + batch_size]


def _valid_mask_from_chip(
    chip: np.ndarray,
    nodata_value: float | None,
    nodata_rule: str,
    control_band_1based: int,
) -> np.ndarray:
    """
    chip: (C,H,W), returns uint8 mask (H,W) with values {0,1}
    """
    h, w = chip.shape[1], chip.shape[2]
    if nodata_value is None:
        return np.ones((h, w), dtype=np.uint8)

    rule = str(nodata_rule or "control-band").strip().lower()
    if rule == "all-bands":
        is_nodata = np.all(chip == float(nodata_value), axis=0)
    else:
        b = int(control_band_1based) - 1
        b = max(0, min(b, chip.shape[0] - 1))
        is_nodata = (chip[b] == float(nodata_value))
    return (~is_nodata).astype(np.uint8)


def _dilate_8(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    return (
        p[1:-1, 1:-1]
        | p[:-2, 1:-1]
        | p[2:, 1:-1]
        | p[1:-1, :-2]
        | p[1:-1, 2:]
        | p[:-2, :-2]
        | p[:-2, 2:]
        | p[2:, :-2]
        | p[2:, 2:]
    )


def _invalid_edge_band(valid_mask: np.ndarray, radius_px: int) -> np.ndarray:
    """
    Build a narrow valid-side band around invalid pixels.
    Returns bool mask on valid pixels within `radius_px` from invalid area.
    """
    radius = int(radius_px)
    if radius <= 0:
        return np.zeros_like(valid_mask, dtype=bool)

    invalid = (valid_mask == 0)
    if not np.any(invalid):
        return np.zeros_like(valid_mask, dtype=bool)

    grown = invalid.copy()
    for _ in range(radius):
        grown = _dilate_8(grown)
    return (valid_mask == 1) & grown


def _validate_scale_01(name: str, value: float) -> float:
    v = float(value)
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name} must be in [0,1], got {value}")
    return v


def _prepare_chip(
    ds: rasterio.DatasetReader,
    tile: TileWindow,
    model_window_size: int,
    norm_stats: NormalizationStats,
    num_bands: int,
    add_valid_channel: bool,
    nodata_value: float | None,
    nodata_rule: str,
    control_band_1based: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    if ds.count < int(num_bands):
        raise RuntimeError(f"{ds.name}: expected at least {int(num_bands)} bands, got {ds.count}")

    indexes = list(range(1, int(num_bands) + 1))
    arr = ds.read(indexes=indexes, window=Window(tile.x, tile.y, tile.w, tile.h)).astype(np.float32)
    _, h, w = arr.shape

    valid = _valid_mask_from_chip(
        arr,
        nodata_value=nodata_value,
        nodata_rule=nodata_rule,
        control_band_1based=control_band_1based,
    )

    if h != model_window_size or w != model_window_size:
        pad = np.zeros((arr.shape[0], model_window_size, model_window_size), dtype=np.float32)
        pad[:, :h, :w] = arr
        arr = pad

        valid_pad = np.zeros((model_window_size, model_window_size), dtype=np.uint8)
        valid_pad[:h, :w] = valid
        valid = valid_pad

    arr = normalize_image(arr, norm_stats, nodata_value=nodata_value, valid_mask=valid)
    if add_valid_channel:
        arr = np.concatenate([arr, valid[None, :, :].astype(np.float32)], axis=0)

    return arr, valid, (h, w)


@torch.no_grad()
def predict_aoi_raster(
    model: torch.nn.Module,
    plan: RuntimePlan,
    aoi_raster_path: Path,
    norm_stats: NormalizationStats,
    out_extent_path: Path,
    out_boundary_path: Path,
    window_size: int,
    stride: int,
    batch_size: int,
    blend: str = "mean",
    gaussian_sigma: float = 0.30,
    gaussian_min_weight: float = 0.05,
    out_dtype: str = "float32",
    compress: str = "DEFLATE",
    tiled: bool = True,
    bigtiff: str = "if_needed",
    num_bands: int = 8,
    add_valid_channel: bool = True,
    nodata_value: float | None = 65536.0,
    nodata_rule: str = "control-band",
    control_band_1based: int = 1,
    invalid_edge_guard_px: int = 0,
    invalid_edge_extent_scale: float = 1.0,
    invalid_edge_boundary_scale: float = 1.0,
) -> Dict[str, object]:
    window_size = int(window_size)
    stride = int(stride)
    batch_size = int(batch_size)

    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}")
    if stride > window_size:
        raise ValueError(f"stride must be <= window_size to avoid coverage gaps, got stride={stride}, window_size={window_size}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    guard_px = int(invalid_edge_guard_px)
    if guard_px < 0:
        raise ValueError(f"invalid_edge_guard_px must be >= 0, got {invalid_edge_guard_px}")
    guard_extent_scale = _validate_scale_01("invalid_edge_extent_scale", invalid_edge_extent_scale)
    guard_boundary_scale = _validate_scale_01("invalid_edge_boundary_scale", invalid_edge_boundary_scale)

    model.eval()

    with rasterio.open(aoi_raster_path) as ds:
        height = int(ds.height)
        width = int(ds.width)

        tiles = generate_windows(width=width, height=height, window_size=window_size, stride=stride)

        acc_extent = np.zeros((height, width), dtype=np.float32)
        acc_boundary = np.zeros((height, width), dtype=np.float32)
        acc_weight = np.zeros((height, width), dtype=np.float32)
        weight_cache: dict[tuple[int, int], np.ndarray] = {}
        guard_applied_pixels = 0

        device = torch.device(plan.device)

        for group in _batch_iter(tiles, batch_size):
            chips = []
            valid_masks = []
            valid_hw = []
            for tile in group:
                chip, valid_mask, hw = _prepare_chip(
                    ds=ds,
                    tile=tile,
                    model_window_size=window_size,
                    norm_stats=norm_stats,
                    num_bands=num_bands,
                    add_valid_channel=add_valid_channel,
                    nodata_value=nodata_value,
                    nodata_rule=nodata_rule,
                    control_band_1based=control_band_1based,
                )
                chips.append(chip)
                valid_masks.append(valid_mask)
                valid_hw.append(hw)

            x = torch.from_numpy(np.stack(chips, axis=0)).to(device=device, dtype=torch.float32)

            if plan.device == "cuda" and plan.amp_enabled:
                with torch.autocast(device_type="cuda", dtype=amp_dtype_from_plan(plan)):
                    out = model(x)
            else:
                out = model(x)

            # numpy does not support torch bfloat16 directly; cast to float32 first.
            extent_prob = (
                torch.sigmoid(out["extent_logits"])
                .squeeze(1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            boundary_prob = (
                torch.sigmoid(out["boundary_logits"])
                .squeeze(1)
                .detach()
                .float()
                .cpu()
                .numpy()
            )

            for i, tile in enumerate(group):
                h, w = valid_hw[i]
                e = extent_prob[i, :h, :w]
                b = boundary_prob[i, :h, :w]
                valid = valid_masks[i][:h, :w]

                # NoData suppression guard-rail
                e[valid == 0] = 0.0
                b[valid == 0] = 0.0

                if guard_px > 0 and (guard_extent_scale < 1.0 or guard_boundary_scale < 1.0):
                    edge_band = _invalid_edge_band(valid, radius_px=guard_px)
                    if np.any(edge_band):
                        guard_applied_pixels += int(edge_band.sum())
                        if guard_extent_scale < 1.0:
                            e[edge_band] *= guard_extent_scale
                        if guard_boundary_scale < 1.0:
                            b[edge_band] *= guard_boundary_scale

                ww = weight_cache.get((h, w))
                if ww is None:
                    ww = blend_weights(
                        h=h,
                        w=w,
                        mode=blend,
                        gaussian_sigma=gaussian_sigma,
                        gaussian_min_weight=gaussian_min_weight,
                    )
                    weight_cache[(h, w)] = ww

                y0, y1 = tile.y, tile.y + h
                x0, x1 = tile.x, tile.x + w

                acc_extent[y0:y1, x0:x1] += e * ww
                acc_boundary[y0:y1, x0:x1] += b * ww
                acc_weight[y0:y1, x0:x1] += ww

        denom = np.clip(acc_weight, 1e-6, None)
        extent_final = acc_extent / denom
        boundary_final = acc_boundary / denom

        dtype = str(out_dtype).lower()
        if dtype not in {"float32", "float16"}:
            raise ValueError(f"Unsupported output dtype: {out_dtype}")

        meta = ds.meta.copy()
        meta.update(
            {
                "count": 1,
                "dtype": dtype,
                "compress": compress,
                "tiled": bool(tiled),
            }
        )
        meta.pop("nodata", None)
        bt = str(bigtiff).lower()
        if bt in {"yes", "no", "if_needed", "if_safer"}:
            meta["BIGTIFF"] = bt.upper()

        out_extent_path.parent.mkdir(parents=True, exist_ok=True)
        out_boundary_path.parent.mkdir(parents=True, exist_ok=True)

        np_dtype = np.float16 if dtype == "float16" else np.float32

        with rasterio.open(out_extent_path, "w", **meta) as out:
            out.write(extent_final.astype(np_dtype), 1)

        with rasterio.open(out_boundary_path, "w", **meta) as out:
            out.write(boundary_final.astype(np_dtype), 1)

    return {
        "aoi_raster": str(aoi_raster_path),
        "extent_prob": str(out_extent_path),
        "boundary_prob": str(out_boundary_path),
        "tiles": len(tiles),
        "window_size": int(window_size),
        "stride": int(stride),
        "blend": blend,
        "gaussian_sigma": float(gaussian_sigma),
        "gaussian_min_weight": float(gaussian_min_weight),
        "invalid_edge_guard_px": int(guard_px),
        "invalid_edge_extent_scale": float(guard_extent_scale),
        "invalid_edge_boundary_scale": float(guard_boundary_scale),
        "invalid_edge_guard_applied_pixels": int(guard_applied_pixels),
        "overlap_px": int(max(0, window_size - stride)),
        "overlap_ratio": float(max(0.0, min(1.0, (window_size - stride) / float(window_size)))),
    }
