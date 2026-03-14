from __future__ import annotations

import inspect
from typing import Dict

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import sobel
from skimage.segmentation import watershed


_HAS_RSH_MAX_SIZE = "max_size" in inspect.signature(morphology.remove_small_holes).parameters


def _remove_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    if area_threshold <= 0:
        return mask
    if _HAS_RSH_MAX_SIZE:
        return morphology.remove_small_holes(mask, max_size=max(0, int(area_threshold) - 1))
    return morphology.remove_small_holes(mask, area_threshold=int(area_threshold))


def split_fields(
    field_mask: np.ndarray,
    extent_smooth: np.ndarray,
    boundary_smooth: np.ndarray,
    markers: np.ndarray,
    use_watershed: bool,
    boundary_weight: float,
    sobel_weight: float = 0.0,
) -> np.ndarray:
    """Split touching field blobs into instance labels."""
    mask = field_mask.astype(bool)

    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.int32)

    if not use_watershed:
        labels, _ = ndi.label(mask)
        return labels.astype(np.int32)

    if int(markers.max()) <= 0:
        markers, _ = ndi.label(mask)

    # Lower energy = easier flood. Boundary probability creates barrier.
    energy = (1.0 - extent_smooth.astype(np.float32, copy=False))
    energy += float(boundary_weight) * boundary_smooth.astype(np.float32, copy=False)

    if float(sobel_weight) > 0:
        grad = sobel(extent_smooth.astype(np.float32, copy=False))
        energy += float(sobel_weight) * grad

    labels = watershed(energy, markers=markers.astype(np.int32), mask=mask, watershed_line=True)
    return labels.astype(np.int32)


def _relabel_consecutive(labels: np.ndarray) -> np.ndarray:
    out = np.zeros_like(labels, dtype=np.int32)
    uniq = np.unique(labels)
    uniq = uniq[uniq > 0]
    for new_id, old_id in enumerate(uniq.tolist(), start=1):
        out[labels == old_id] = new_id
    return out


def _fill_small_holes_per_label(labels: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    if max_hole_area_px <= 0:
        return labels

    out = labels.copy()
    uniq = np.unique(out)
    uniq = uniq[uniq > 0]

    for lbl in uniq:
        mask = out == lbl
        if not np.any(mask):
            continue
        filled = _remove_small_holes(mask, area_threshold=int(max_hole_area_px))
        add = filled & (~mask) & (out == 0)
        out[add] = int(lbl)

    return out


def _merge_small_regions(labels: np.ndarray, max_area_px: int) -> np.ndarray:
    if max_area_px <= 0:
        return labels

    out = labels.copy()
    sizes = np.bincount(out.ravel())
    small = np.where((sizes > 0) & (sizes <= int(max_area_px)))[0]
    small = small[small > 0]

    if small.size == 0:
        return out

    # Small-first strategy to reduce fragmented islands.
    small_sorted = sorted(small.tolist(), key=lambda x: sizes[x])

    for lbl in small_sorted:
        region = out == lbl
        if not np.any(region):
            continue

        border = ndi.binary_dilation(region, structure=np.ones((3, 3), dtype=bool)) & (~region)
        neigh = out[border]
        neigh = neigh[(neigh > 0) & (neigh != lbl)]

        if neigh.size == 0:
            out[region] = 0
            continue

        counts = np.bincount(neigh)
        target = int(np.argmax(counts))
        if target <= 0:
            out[region] = 0
        else:
            out[region] = target

    return out


def _drop_tiny_regions(labels: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return labels

    out = labels.copy()
    sizes = np.bincount(out.ravel())
    tiny = np.where((sizes > 0) & (sizes < int(min_area_px)))[0]
    tiny = tiny[tiny > 0]
    if tiny.size == 0:
        return out

    for lbl in tiny.tolist():
        out[out == lbl] = 0
    return out


def clean_labels(
    labels: np.ndarray,
    min_region_area_px: int,
    fill_holes_max_area_px: int,
    small_region_max_area_px: int,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """
    Clean raster labels:
    1) fill small holes inside each object;
    2) merge tiny fragments into neighboring labels;
    3) drop very small leftovers;
    4) relabel to consecutive IDs.
    """
    out = labels.astype(np.int32, copy=True)
    out[~valid_mask.astype(bool)] = 0

    out = _fill_small_holes_per_label(out, int(fill_holes_max_area_px))
    out = _merge_small_regions(out, int(small_region_max_area_px))
    out = _drop_tiny_regions(out, int(min_region_area_px))

    out[~valid_mask.astype(bool)] = 0
    out = _relabel_consecutive(out)
    return out.astype(np.int32)


def labels_stats(labels: np.ndarray) -> Dict[str, int]:
    uniq = np.unique(labels)
    uniq = uniq[uniq > 0]
    return {
        "num_labels": int(uniq.size),
        "max_label": int(labels.max()) if labels.size > 0 else 0,
        "num_pixels_fg": int((labels > 0).sum()),
    }
