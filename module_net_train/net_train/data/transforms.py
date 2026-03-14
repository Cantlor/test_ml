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
    invalid_edge_sim_enabled: bool = False
    invalid_edge_sim_prob: float = 0.0
    invalid_edge_sim_min_width_px: int = 8
    invalid_edge_sim_max_width_px: int = 64
    invalid_edge_sim_block_prob: float = 0.35
    invalid_edge_sim_max_area_ratio: float = 0.20
    invalid_edge_sim_zero_image: bool = True


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


def near_invalid_band(valid_mask: np.ndarray, radius_px: int = 2) -> np.ndarray:
    """
    Returns bool mask over valid pixels that are within `radius_px` from invalid pixels.
    """
    radius = max(0, int(radius_px))
    if radius == 0:
        return np.zeros_like(valid_mask, dtype=bool)

    invalid = (valid_mask == 0)
    if not np.any(invalid):
        return np.zeros_like(valid_mask, dtype=bool)

    grown = invalid.copy()
    for _ in range(radius):
        grown = _dilate_8(grown)
    return (valid_mask == 1) & grown



def random_crop(
    img: np.ndarray,
    extent: np.ndarray,
    boundary: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
    min_extent_pixels: int = 0,
    min_boundary_pixels: int = 0,
    attempts: int = 1,
    fallback_to_best_prob: float = 1.0,
    valid_mask: np.ndarray | None = None,
    near_invalid_radius_px: int = 2,
    min_near_invalid_pixels: int = 0,
    near_invalid_bias_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random crop for image+targets. img is (C,H,W), targets are (H,W)."""
    _, h, w = img.shape
    if crop_size <= 0 or crop_size >= h or crop_size >= w:
        return img, extent, boundary

    min_extent_pixels = max(0, int(min_extent_pixels))
    min_boundary_pixels = max(0, int(min_boundary_pixels))
    attempts = max(1, int(attempts))
    fallback_to_best_prob = float(np.clip(float(fallback_to_best_prob), 0.0, 1.0))
    min_near_invalid_pixels = max(0, int(min_near_invalid_pixels))
    near_invalid_bias_prob = float(np.clip(float(near_invalid_bias_prob), 0.0, 1.0))

    near_map: np.ndarray | None = None
    enforce_near_invalid = False
    if valid_mask is not None and min_near_invalid_pixels > 0 and near_invalid_bias_prob > 0.0:
        near_map = near_invalid_band(valid_mask, radius_px=int(near_invalid_radius_px))
        if np.any(near_map):
            enforce_near_invalid = float(rng.random()) < near_invalid_bias_prob

    best = None
    best_score = (-1, -1, -1)

    for _ in range(attempts):
        y0 = int(rng.integers(0, h - crop_size + 1))
        x0 = int(rng.integers(0, w - crop_size + 1))
        e = extent[y0:y0 + crop_size, x0:x0 + crop_size]
        b = boundary[y0:y0 + crop_size, x0:x0 + crop_size]
        n_edge = (
            int(near_map[y0:y0 + crop_size, x0:x0 + crop_size].sum())
            if near_map is not None
            else 0
        )

        e_pos = int((e == 1).sum())
        b_pos = int((b == 1).sum())
        ok = e_pos >= min_extent_pixels and b_pos >= min_boundary_pixels
        if enforce_near_invalid:
            ok = ok and (n_edge >= min_near_invalid_pixels)
        score = (n_edge, b_pos, e_pos) if enforce_near_invalid else (b_pos, e_pos, n_edge)
        if ok:
            return (
                img[:, y0:y0 + crop_size, x0:x0 + crop_size],
                e,
                b,
            )
        if score > best_score:
            best = (y0, x0, e, b)
            best_score = score

    if best is None:
        y0 = int(rng.integers(0, h - crop_size + 1))
        x0 = int(rng.integers(0, w - crop_size + 1))
        return (
            img[:, y0:y0 + crop_size, x0:x0 + crop_size],
            extent[y0:y0 + crop_size, x0:x0 + crop_size],
            boundary[y0:y0 + crop_size, x0:x0 + crop_size],
        )

    if float(rng.random()) < fallback_to_best_prob:
        y0, x0, e, b = best
        return (
            img[:, y0:y0 + crop_size, x0:x0 + crop_size],
            e,
            b,
        )

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
    def __init__(
        self,
        cfg: AugmentConfig,
        seed: int = 123,
        valid_channel_index: Optional[int] = None,
    ):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.valid_channel_index = valid_channel_index
        self.last_invalid_edge_applied: bool = False

    def _sample_invalid_mask(self, h: int, w: int) -> np.ndarray:
        min_w = max(1, int(self.cfg.invalid_edge_sim_min_width_px))
        max_w = max(min_w, int(self.cfg.invalid_edge_sim_max_width_px))
        max_area_ratio = float(np.clip(float(self.cfg.invalid_edge_sim_max_area_ratio), 0.01, 0.95))
        max_area = max(1, int(max_area_ratio * h * w))

        mask = np.zeros((h, w), dtype=bool)
        do_block = float(self.rng.random()) < float(np.clip(self.cfg.invalid_edge_sim_block_prob, 0.0, 1.0))
        if do_block:
            bw = int(self.rng.integers(min_w, max_w + 1))
            bh = int(self.rng.integers(min_w, max_w + 1))
            bw = min(max(1, bw), w)
            bh = min(max(1, bh), h)
            if bh * bw > max_area:
                scale = float(np.sqrt(max_area / float(bh * bw)))
                bh = max(1, int(bh * scale))
                bw = max(1, int(bw * scale))
            y0 = int(self.rng.integers(0, max(1, h - bh + 1)))
            x0 = int(self.rng.integers(0, max(1, w - bw + 1)))
            mask[y0:y0 + bh, x0:x0 + bw] = True
            return mask

        side = int(self.rng.integers(0, 4))  # 0: left, 1: right, 2: top, 3: bottom
        if side in (0, 1):
            ww = int(self.rng.integers(min_w, max_w + 1))
            ww = min(max(1, ww), max(1, w - 1))
            if h * ww > max_area:
                ww = max(1, int(max_area / max(1, h)))
            if side == 0:
                mask[:, :ww] = True
            else:
                mask[:, w - ww:] = True
        else:
            hh = int(self.rng.integers(min_w, max_w + 1))
            hh = min(max(1, hh), max(1, h - 1))
            if w * hh > max_area:
                hh = max(1, int(max_area / max(1, w)))
            if side == 2:
                mask[:hh, :] = True
            else:
                mask[h - hh:, :] = True

        return mask

    def _apply_invalid_edge_sim(self, img: np.ndarray) -> tuple[np.ndarray, bool]:
        self.last_invalid_edge_applied = False

        if not self.cfg.invalid_edge_sim_enabled:
            return img, False
        if self.valid_channel_index is None:
            return img, False
        if float(self.rng.random()) >= float(np.clip(self.cfg.invalid_edge_sim_prob, 0.0, 1.0)):
            return img, False

        vi = int(self.valid_channel_index)
        if vi < 0 or vi >= img.shape[0]:
            return img, False

        _, h, w = img.shape
        mask = self._sample_invalid_mask(h=h, w=w)
        if not np.any(mask):
            return img, False

        if bool(self.cfg.invalid_edge_sim_zero_image) and vi > 0:
            img[:vi, mask] = 0.0
        img[vi, mask] = 0.0
        self.last_invalid_edge_applied = True
        return img, True

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

        img, _ = self._apply_invalid_edge_sim(img)

        # np.rot90 returns view with negative strides -> force contiguous
        return np.ascontiguousarray(img), np.ascontiguousarray(extent), np.ascontiguousarray(boundary)
