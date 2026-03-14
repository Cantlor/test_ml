from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TileWindow:
    x: int
    y: int
    w: int
    h: int



def _starts(total: int, win: int, stride: int) -> List[int]:
    if total <= win:
        return [0]

    vals = list(range(0, total - win + 1, stride))
    if vals[-1] != total - win:
        vals.append(total - win)
    return vals



def generate_windows(width: int, height: int, window_size: int, stride: int) -> List[TileWindow]:
    w = int(min(window_size, width))
    h = int(min(window_size, height))

    xs = _starts(width, w, stride)
    ys = _starts(height, h, stride)

    out: List[TileWindow] = []
    for y in ys:
        for x in xs:
            out.append(TileWindow(x=x, y=y, w=w, h=h))
    return out



def blend_weights(
    h: int,
    w: int,
    mode: str = "mean",
    gaussian_sigma: float = 0.30,
    gaussian_min_weight: float = 0.05,
) -> np.ndarray:
    mode = mode.lower()
    if mode == "mean":
        return np.ones((h, w), dtype=np.float32)

    if mode == "gaussian":
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        yv, xv = np.meshgrid(yy, xx, indexing="ij")
        rr2 = xv * xv + yv * yv
        sigma = float(gaussian_sigma)
        if sigma <= 0.0:
            raise ValueError(f"gaussian_sigma must be > 0, got {gaussian_sigma}")
        sigma2 = sigma * sigma
        g = np.exp(-0.5 * rr2 / sigma2).astype(np.float32)
        g /= np.clip(g.max(), 1e-6, None)
        min_w = float(gaussian_min_weight)
        if min_w < 0.0:
            raise ValueError(f"gaussian_min_weight must be >= 0, got {gaussian_min_weight}")
        if min_w > 0.0:
            g = np.maximum(g, min_w).astype(np.float32)
        return g

    raise ValueError(f"Unknown blend mode: {mode}")
