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



def _sample_pixels(img: np.ndarray, max_pixels: int, rng: np.random.Generator) -> np.ndarray:
    """img: (C,H,W) -> sampled (C,N)."""
    c, h, w = img.shape
    flat = img.reshape(c, h * w)
    n = flat.shape[1]
    if n <= max_pixels:
        return flat
    idx = rng.choice(n, size=max_pixels, replace=False)
    return flat[:, idx]



def _filter_nodata(sample: np.ndarray, nodata_value: float | None, ignore_nodata: bool) -> np.ndarray:
    if nodata_value is None or not ignore_nodata:
        return sample

    valid = np.ones(sample.shape[1], dtype=bool)
    for c in range(sample.shape[0]):
        valid &= sample[c] != nodata_value
    if not valid.any():
        return sample
    return sample[:, valid]



def compute_normalization_stats(
    records: List[SampleRecord],
    mode: str,
    per_band: bool,
    nodata_value: float | None,
    ignore_nodata: bool,
    p_low: float = 2.0,
    p_high: float = 98.0,
    max_pixels_per_image: int = 25000,
    seed: int = 123,
) -> NormalizationStats:
    rng = np.random.default_rng(seed)

    if not records:
        raise RuntimeError("Cannot compute normalization stats from empty record list")

    samples = []
    for rec in records:
        with rasterio.open(rec.img_path) as ds:
            img = ds.read().astype(np.float32)
        sampled = _sample_pixels(img, max_pixels=max_pixels_per_image, rng=rng)
        sampled = _filter_nodata(sampled, nodata_value=nodata_value, ignore_nodata=ignore_nodata)
        if sampled.size > 0:
            samples.append(sampled)

    if not samples:
        raise RuntimeError("No valid pixels for normalization stats")

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
) -> np.ndarray:
    """Normalize image (C,H,W), return float32."""
    x = img.astype(np.float32, copy=False)
    nodata_mask = None
    if nodata_value is not None:
        nodata_mask = np.all(x == float(nodata_value), axis=0)

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
        if nodata_mask is not None and nodata_mask.any():
            x[:, nodata_mask] = 0.0
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
        if nodata_mask is not None and nodata_mask.any():
            x[:, nodata_mask] = 0.0
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
