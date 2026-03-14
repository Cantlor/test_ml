from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import morphology


def build_markers(
    field_mask: np.ndarray,
    boundary_barrier: np.ndarray,
    seed_min_distance_px: int,
    seed_hmax: float,
    marker_erode_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build marker raster for watershed using distance transform + local maxima.

    Returns:
        markers: int32 labels for watershed seeds
        distance: float32 EDT map used for diagnostics
    """
    core = field_mask.astype(bool) & (~boundary_barrier.astype(bool))

    if marker_erode_px > 0:
        core = morphology.erosion(core, footprint=morphology.disk(int(marker_erode_px)))

    if not np.any(core):
        core = field_mask.astype(bool)

    distance = ndi.distance_transform_edt(core).astype(np.float32)

    seeds_mask = np.zeros_like(core, dtype=bool)

    if float(seed_hmax) > 0:
        hmax_mask = morphology.h_maxima(distance, float(seed_hmax)).astype(bool)
        seeds_mask |= (hmax_mask & core)

    coords = peak_local_max(
        distance,
        min_distance=max(1, int(seed_min_distance_px)),
        labels=core.astype(np.uint8),
        exclude_border=False,
    )
    if coords.size > 0:
        seeds_mask[coords[:, 0], coords[:, 1]] = True

    seeds_mask &= core

    if not np.any(seeds_mask):
        markers, count = ndi.label(core)
        if count == 0:
            markers, _ = ndi.label(field_mask.astype(bool))
        return markers.astype(np.int32), distance

    markers, count = ndi.label(seeds_mask)
    if count == 0:
        markers, _ = ndi.label(core)

    return markers.astype(np.int32), distance
