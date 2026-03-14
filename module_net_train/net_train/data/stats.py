from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio

from net_train.data.index import SampleRecord


@dataclass
class NormalizationStats:
    mode: str
    per_band: bool
    p_low: float | None
    p_high: float | None
    q_low: List[float] | None
    q_high: List[float] | None
    mean: List[float] | None
    std: List[float] | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sample_pixels_with_valid(
    img: np.ndarray,
    valid_u8: np.ndarray,
    max_pixels: int,
    rng: np.random.Generator,
    ignore_nodata: bool,
) -> np.ndarray:
    """
    img: (C,H,W), valid_u8: (H,W) with {0,1}
    returns sampled pixels as (C,N)
    """
    c, h, w = img.shape
    flat = img.reshape(c, h * w)
    valid_flat = (valid_u8.reshape(h * w) > 0)

    if ignore_nodata:
        idx_all = np.flatnonzero(valid_flat)
    else:
        idx_all = np.arange(h * w, dtype=np.int64)

    if idx_all.size == 0:
        return np.empty((c, 0), dtype=img.dtype)

    if idx_all.size > max_pixels:
        idx = rng.choice(idx_all, size=max_pixels, replace=False)
        return flat[:, idx]
    return flat[:, idx_all]



def compute_normalization_stats(
    records: List[SampleRecord],
    mode: str,
    per_band: bool,
    nodata_value: float | None,
    ignore_nodata: bool,
    nodata_rule: str = "control-band",
    control_band_1based: int = 1,
    p_low: float = 2.0,
    p_high: float = 98.0,
    max_pixels_per_image: int = 25000,
    seed: int = 123,
    image_bands: int = 8,
) -> NormalizationStats:
    rng = np.random.default_rng(seed)

    if not records:
        raise RuntimeError("Cannot compute normalization stats from empty record list")

    samples = []
    rule = str(nodata_rule or "control-band").strip().lower()
    cb = max(0, int(control_band_1based) - 1)

    for rec in records:
        with rasterio.open(rec.img_path) as ds:
            img = ds.read().astype(np.float32)

        if img.shape[0] < int(image_bands):
            raise RuntimeError(
                f"{rec.img_path}: expected at least {int(image_bands)} bands for normalization stats, got {img.shape[0]}"
            )
        img = img[: int(image_bands)]

        valid = np.ones((img.shape[1], img.shape[2]), dtype=np.uint8)
        valid_path = getattr(rec, "valid_path", None)
        if valid_path is not None:
            with rasterio.open(valid_path) as ds:
                valid = (ds.read(1) > 0).astype(np.uint8)
        elif nodata_value is not None:
            if rule == "all-bands":
                nodata_mask = np.all(img == float(nodata_value), axis=0)
            else:
                b = max(0, min(cb, img.shape[0] - 1))
                nodata_mask = (img[b] == float(nodata_value))
            valid = (~nodata_mask).astype(np.uint8)

        sampled = _sample_pixels_with_valid(
            img=img,
            valid_u8=valid,
            max_pixels=max_pixels_per_image,
            rng=rng,
            ignore_nodata=bool(ignore_nodata),
        )
        if sampled.shape[1] > 0:
            samples.append(sampled)

    if not samples:
        raise RuntimeError("No valid pixels for normalization stats (check valid masks / nodata policy)")

    stacked = np.concatenate(samples, axis=1)  # (C,N)

    if mode == "robust_percentile":
        if per_band:
            ql = np.percentile(stacked, p_low, axis=1).astype(np.float32)
            qh = np.percentile(stacked, p_high, axis=1).astype(np.float32)
        else:
            lo = float(np.percentile(stacked, p_low))
            hi = float(np.percentile(stacked, p_high))
            ql = np.array([lo], dtype=np.float32)
            qh = np.array([hi], dtype=np.float32)

        return NormalizationStats(
            mode=mode,
            per_band=per_band,
            p_low=float(p_low),
            p_high=float(p_high),
            q_low=[float(v) for v in ql.tolist()],
            q_high=[float(v) for v in qh.tolist()],
            mean=None,
            std=None,
        )

    if mode == "mean_std":
        if per_band:
            mu = stacked.mean(axis=1).astype(np.float32)
            sd = stacked.std(axis=1).astype(np.float32)
        else:
            mu_val = float(stacked.mean())
            sd_val = float(stacked.std())
            mu = np.array([mu_val], dtype=np.float32)
            sd = np.array([sd_val], dtype=np.float32)

        sd = np.clip(sd, 1e-6, None)

        return NormalizationStats(
            mode=mode,
            per_band=per_band,
            p_low=None,
            p_high=None,
            q_low=None,
            q_high=None,
            mean=[float(v) for v in mu.tolist()],
            std=[float(v) for v in sd.tolist()],
        )

    raise ValueError(f"Unknown normalization mode: {mode}")



def normalize_image(
    img: np.ndarray,
    stats: NormalizationStats,
    nodata_value: float | None = None,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize image (C,H,W), return float32."""
    x = img.astype(np.float32, copy=False)
    invalid_mask = None
    if valid_mask is not None:
        invalid_mask = (valid_mask <= 0)
    elif nodata_value is not None:
        invalid_mask = np.all(x == float(nodata_value), axis=0)

    if stats.mode == "robust_percentile":
        ql = np.asarray(stats.q_low, dtype=np.float32)
        qh = np.asarray(stats.q_high, dtype=np.float32)

        if stats.per_band:
            ql = ql.reshape(-1, 1, 1)
            qh = qh.reshape(-1, 1, 1)
        else:
            ql = ql.reshape(1, 1, 1)
            qh = qh.reshape(1, 1, 1)

        denom = np.clip(qh - ql, 1e-6, None)
        x = (x - ql) / denom
        x = np.clip(x, 0.0, 1.0)
        if invalid_mask is not None and invalid_mask.any():
            x[:, invalid_mask] = 0.0
        return x

    if stats.mode == "mean_std":
        mu = np.asarray(stats.mean, dtype=np.float32)
        sd = np.asarray(stats.std, dtype=np.float32)
        if stats.per_band:
            mu = mu.reshape(-1, 1, 1)
            sd = sd.reshape(-1, 1, 1)
        else:
            mu = mu.reshape(1, 1, 1)
            sd = sd.reshape(1, 1, 1)
        x = (x - mu) / np.clip(sd, 1e-6, None)
        if invalid_mask is not None and invalid_mask.any():
            x[:, invalid_mask] = 0.0
        return x

    raise ValueError(f"Unknown normalization mode: {stats.mode}")



def save_stats_npz(path: Path, stats: NormalizationStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **stats.to_dict())



def load_stats_npz(path: Path) -> NormalizationStats:
    arr = np.load(path, allow_pickle=True)
    data = {k: arr[k].tolist() for k in arr.files}

    return NormalizationStats(
        mode=str(data["mode"]),
        per_band=bool(data["per_band"]),
        p_low=None if data.get("p_low") is None else float(data.get("p_low")),
        p_high=None if data.get("p_high") is None else float(data.get("p_high")),
        q_low=None if data.get("q_low") is None else [float(v) for v in data.get("q_low")],
        q_high=None if data.get("q_high") is None else [float(v) for v in data.get("q_high")],
        mean=None if data.get("mean") is None else [float(v) for v in data.get("mean")],
        std=None if data.get("std") is None else [float(v) for v in data.get("std")],
    )
