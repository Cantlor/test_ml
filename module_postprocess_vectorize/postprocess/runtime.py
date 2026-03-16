from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, Mapping, Optional

import numpy as np


_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RuntimePolicy:
    n_pixels: int
    valid_pixels: int
    cpu_count: int
    available_ram_bytes: int
    ram_budget_bytes: int
    estimated_peak_bytes: int
    estimated_pressure: float
    prob_dtype: str
    smooth_dtype: str
    gaussian_sigma_px_effective: float
    use_watershed: bool
    sobel_weight_effective: float
    clean_labels_mode: str
    warn_large_raster: bool
    warnings: list[str]

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def detect_available_ram_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().available)
    except Exception:
        pass

    meminfo = "/proc/meminfo"
    try:
        with open(meminfo, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and available_pages > 0:
            return page_size * available_pages
    except Exception:
        pass

    # Conservative fallback when runtime probing is unavailable.
    return 8 * 1024 * 1024 * 1024


def detect_cpu_count() -> int:
    n = os.cpu_count()
    if n is None or n < 1:
        return 1
    return int(n)


def _dtype_nbytes(dtype_name: str, *, fallback: str) -> int:
    try:
        return int(np.dtype(dtype_name).itemsize)
    except Exception:
        return int(np.dtype(fallback).itemsize)


def _estimate_peak_bytes(
    *,
    n_pixels: int,
    prob_dtype: str,
    smooth_dtype: str,
    use_watershed: bool,
    clean_labels_mode: str,
) -> int:
    prob_b = _dtype_nbytes(prob_dtype, fallback="float16")
    smooth_b = _dtype_nbytes(smooth_dtype, fallback="float32")

    # Full-frame arrays used across the raster pipeline.
    bytes_per_px = (
        (2 * prob_b)   # extent_prob + boundary_prob
        + (2 * smooth_b)  # smoothed maps
        + 1  # valid mask
        + 2  # field/boundary boolean masks
        + 8  # labels_raw + labels (int32)
    )
    if use_watershed:
        bytes_per_px += 12  # energy + markers + seeds (float32/int32/int32)

    overhead = 2.0 if clean_labels_mode == "exact" else 1.6
    return int(float(n_pixels) * float(bytes_per_px) * float(overhead))


def _get_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except Exception:
        return float(default)


def _get_int(cfg: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, default))
    except Exception:
        return int(default)


def _get_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    try:
        return bool(cfg.get(key, default))
    except Exception:
        return bool(default)


def build_runtime_policy(
    *,
    config: Mapping[str, Any],
    memory_cfg: Mapping[str, Any],
    n_pixels: int,
    valid_pixels: Optional[int],
) -> RuntimePolicy:
    warnings: list[str] = []

    cpu_count = detect_cpu_count()
    available_ram_bytes = max(1, detect_available_ram_bytes())

    ram_budget_fraction = _get_float(memory_cfg, "ram_budget_fraction", 0.30)
    ram_budget_fraction = min(0.90, max(0.05, ram_budget_fraction))
    ram_guard_mb = _get_float(memory_cfg, "ram_guard_mb", 1024.0)
    min_ram_budget_mb = _get_float(memory_cfg, "min_ram_budget_mb", 768.0)
    ram_guard_bytes = int(max(0.0, ram_guard_mb) * 1024.0 * 1024.0)
    min_ram_budget_bytes = int(max(64.0, min_ram_budget_mb) * 1024.0 * 1024.0)

    raw_budget = int(float(available_ram_bytes) * ram_budget_fraction) - ram_guard_bytes
    ram_budget_bytes = max(min_ram_budget_bytes, min(available_ram_bytes, raw_budget))

    prob_dtype = str(memory_cfg.get("prob_dtype", "float16"))
    smooth_dtype_cfg = str(memory_cfg.get("smooth_dtype", "auto")).strip().lower()
    if smooth_dtype_cfg == "auto":
        smooth_dtype = "float32"
    else:
        smooth_dtype = str(memory_cfg.get("smooth_dtype", "float32"))

    smooth_dtype_large = str(memory_cfg.get("smooth_dtype_large", "float16"))

    gaussian_sigma_base = _get_float(config, "gaussian_sigma_px", 1.0)
    gaussian_sigma_px_effective = float(max(0.0, gaussian_sigma_base))

    sobel_weight_default = _get_float(config, "sobel_weight", 0.0)
    sobel_weight_large = _get_float(memory_cfg, "sobel_weight_large", 0.0)
    sobel_weight_effective = float(sobel_weight_default)

    use_watershed_cfg = bool(config.get("use_watershed", True))
    force_no_watershed = _get_bool(memory_cfg, "force_no_watershed", False)
    auto_disable_watershed = _get_bool(memory_cfg, "auto_disable_watershed", True)
    use_watershed = bool(use_watershed_cfg and not force_no_watershed)

    clean_labels_mode_cfg = str(memory_cfg.get("clean_labels_mode", "auto")).strip().lower()
    clean_labels_fast_pixels_threshold = _get_int(memory_cfg, "clean_labels_fast_pixels_threshold", 40_000_000)

    if clean_labels_mode_cfg in {"exact", "fast"}:
        clean_labels_mode = clean_labels_mode_cfg
    else:
        clean_labels_mode = "fast" if n_pixels >= clean_labels_fast_pixels_threshold else "exact"

    max_pixels_for_watershed = _get_int(memory_cfg, "max_pixels_for_watershed", 50_000_000)
    max_pixels_for_gaussian = _get_int(memory_cfg, "max_pixels_for_gaussian", max_pixels_for_watershed)
    warn_pixels_threshold = _get_int(memory_cfg, "warn_pixels_threshold", 30_000_000)

    auto_disable_gaussian_large = _get_bool(memory_cfg, "auto_disable_gaussian_large", True)
    warn_large_raster = bool(n_pixels >= warn_pixels_threshold)

    if auto_disable_gaussian_large and n_pixels > max_pixels_for_gaussian and gaussian_sigma_px_effective > 0.0:
        gaussian_sigma_px_effective = max(0.35, gaussian_sigma_px_effective * 0.60)
        warnings.append("gaussian_sigma_reduced_by_pixel_guard")

    if auto_disable_watershed and n_pixels > max_pixels_for_watershed and use_watershed:
        use_watershed = False
        sobel_weight_effective = float(sobel_weight_large)
        warnings.append("watershed_disabled_by_pixel_guard")

    estimated_peak_bytes = _estimate_peak_bytes(
        n_pixels=n_pixels,
        prob_dtype=prob_dtype,
        smooth_dtype=smooth_dtype,
        use_watershed=use_watershed,
        clean_labels_mode=clean_labels_mode,
    )
    estimated_pressure = float(estimated_peak_bytes) / float(max(1, ram_budget_bytes))

    degrade_gaussian_pressure = _get_float(memory_cfg, "degrade_gaussian_pressure", 0.70)
    disable_gaussian_pressure = _get_float(memory_cfg, "disable_gaussian_pressure", 1.15)
    disable_watershed_pressure = _get_float(memory_cfg, "disable_watershed_pressure", 0.95)
    switch_fast_clean_pressure = _get_float(memory_cfg, "switch_fast_clean_pressure", 0.85)

    if estimated_pressure >= 0.75 and smooth_dtype_cfg == "auto":
        smooth_dtype = smooth_dtype_large
        warnings.append("smooth_dtype_switched_to_large_mode")

    if gaussian_sigma_px_effective > 0.0 and estimated_pressure >= degrade_gaussian_pressure:
        reduced = max(0.35, gaussian_sigma_px_effective * 0.75)
        if reduced < gaussian_sigma_px_effective:
            gaussian_sigma_px_effective = reduced
            warnings.append("gaussian_sigma_reduced_by_ram_pressure")

    if auto_disable_gaussian_large and gaussian_sigma_px_effective > 0.0 and estimated_pressure >= disable_gaussian_pressure:
        gaussian_sigma_px_effective = 0.0
        warnings.append("gaussian_disabled_by_ram_pressure")

    if auto_disable_watershed and use_watershed and estimated_pressure >= disable_watershed_pressure:
        use_watershed = False
        sobel_weight_effective = float(sobel_weight_large)
        warnings.append("watershed_disabled_by_ram_pressure")

    if clean_labels_mode_cfg == "auto" and estimated_pressure >= switch_fast_clean_pressure:
        if clean_labels_mode != "fast":
            clean_labels_mode = "fast"
            warnings.append("clean_labels_fast_mode_by_ram_pressure")

    estimated_peak_bytes = _estimate_peak_bytes(
        n_pixels=n_pixels,
        prob_dtype=prob_dtype,
        smooth_dtype=smooth_dtype,
        use_watershed=use_watershed,
        clean_labels_mode=clean_labels_mode,
    )
    estimated_pressure = float(estimated_peak_bytes) / float(max(1, ram_budget_bytes))

    if warn_large_raster:
        _LOG.warning(
            "Large raster detected: %d pixels (%.2f MP). RAM-aware runtime policy is active.",
            n_pixels,
            n_pixels / 1_000_000.0,
        )

    if estimated_pressure >= 1.0:
        warnings.append("estimated_peak_exceeds_ram_budget")

    resolved_valid_pixels = int(valid_pixels) if valid_pixels is not None else int(n_pixels)
    return RuntimePolicy(
        n_pixels=int(n_pixels),
        valid_pixels=max(0, resolved_valid_pixels),
        cpu_count=int(cpu_count),
        available_ram_bytes=int(available_ram_bytes),
        ram_budget_bytes=int(ram_budget_bytes),
        estimated_peak_bytes=int(estimated_peak_bytes),
        estimated_pressure=float(estimated_pressure),
        prob_dtype=str(prob_dtype),
        smooth_dtype=str(smooth_dtype),
        gaussian_sigma_px_effective=float(gaussian_sigma_px_effective),
        use_watershed=bool(use_watershed),
        sobel_weight_effective=float(sobel_weight_effective),
        clean_labels_mode=str(clean_labels_mode),
        warn_large_raster=bool(warn_large_raster),
        warnings=list(warnings),
    )
