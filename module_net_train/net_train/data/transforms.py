from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AugmentConfig:
    enabled: bool = True
    hflip: bool = True
    vflip: bool = True
    rotate90: bool = True



def random_crop(
    img: np.ndarray,
    extent: np.ndarray,
    boundary: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop for image+targets. img is (C,H,W), targets are (H,W)."""
    _, h, w = img.shape
    if crop_size <= 0 or crop_size >= h or crop_size >= w:
        return img, extent, boundary

    y0 = int(rng.integers(0, h - crop_size + 1))
    x0 = int(rng.integers(0, w - crop_size + 1))

    return (
        img[:, y0:y0 + crop_size, x0:x0 + crop_size],
        extent[y0:y0 + crop_size, x0:x0 + crop_size],
        boundary[y0:y0 + crop_size, x0:x0 + crop_size],
    )



def center_crop_or_pad(
    img: np.ndarray,
    extent: np.ndarray,
    boundary: np.ndarray,
    target_size: int,
    pad_value_img: float = 0.0,
    pad_value_extent: int = 255,
    pad_value_boundary: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Center crop or pad to square target size."""
    c, h, w = img.shape

    if h == target_size and w == target_size:
        return img, extent, boundary

    out_img = np.full((c, target_size, target_size), pad_value_img, dtype=img.dtype)
    out_extent = np.full((target_size, target_size), pad_value_extent, dtype=extent.dtype)
    out_boundary = np.full((target_size, target_size), pad_value_boundary, dtype=boundary.dtype)

    src_y0 = max(0, (h - target_size) // 2)
    src_x0 = max(0, (w - target_size) // 2)
    src_y1 = min(h, src_y0 + target_size)
    src_x1 = min(w, src_x0 + target_size)

    crop_h = src_y1 - src_y0
    crop_w = src_x1 - src_x0

    dst_y0 = max(0, (target_size - crop_h) // 2)
    dst_x0 = max(0, (target_size - crop_w) // 2)

    out_img[:, dst_y0:dst_y0 + crop_h, dst_x0:dst_x0 + crop_w] = img[:, src_y0:src_y1, src_x0:src_x1]
    out_extent[dst_y0:dst_y0 + crop_h, dst_x0:dst_x0 + crop_w] = extent[src_y0:src_y1, src_x0:src_x1]
    out_boundary[dst_y0:dst_y0 + crop_h, dst_x0:dst_x0 + crop_w] = boundary[src_y0:src_y1, src_x0:src_x1]

    return out_img, out_extent, out_boundary


class TrainAugmentor:
    def __init__(self, cfg: AugmentConfig, seed: int = 123):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def __call__(
        self,
        img: np.ndarray,
        extent: np.ndarray,
        boundary: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.cfg.enabled:
            return img, extent, boundary

        if self.cfg.hflip and float(self.rng.random()) < 0.5:
            img = img[:, :, ::-1]
            extent = extent[:, ::-1]
            boundary = boundary[:, ::-1]

        if self.cfg.vflip and float(self.rng.random()) < 0.5:
            img = img[:, ::-1, :]
            extent = extent[::-1, :]
            boundary = boundary[::-1, :]

        if self.cfg.rotate90:
            k = int(self.rng.integers(0, 4))
            if k:
                img = np.rot90(img, k=k, axes=(1, 2))
                extent = np.rot90(extent, k=k, axes=(0, 1))
                boundary = np.rot90(boundary, k=k, axes=(0, 1))

        # np.rot90 returns view with negative strides -> force contiguous
        return np.ascontiguousarray(img), np.ascontiguousarray(extent), np.ascontiguousarray(boundary)
