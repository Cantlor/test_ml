from __future__ import annotations

import logging
import inspect
from typing import Tuple

import numpy as np
from affine import Affine
from pyproj import CRS as PyCRS
from pyproj import Geod, Transformer
from rasterio.crs import CRS
from scipy import ndimage as ndi
from skimage import morphology


_LOG = logging.getLogger(__name__)
_HAS_RSO_MAX_SIZE = "max_size" in inspect.signature(morphology.remove_small_objects).parameters
_HAS_RSH_MAX_SIZE = "max_size" in inspect.signature(morphology.remove_small_holes).parameters


def _remove_small_objects(mask: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return mask
    # skimage>=0.26 deprecates min_size in favor of max_size with <= semantics.
    if _HAS_RSO_MAX_SIZE:
        return morphology.remove_small_objects(mask, max_size=max(0, int(min_size) - 1))
    return morphology.remove_small_objects(mask, min_size=int(min_size))


def _remove_small_holes(mask: np.ndarray, area_threshold: int) -> np.ndarray:
    if area_threshold <= 0:
        return mask
    if _HAS_RSH_MAX_SIZE:
        return morphology.remove_small_holes(mask, max_size=max(0, int(area_threshold) - 1))
    return morphology.remove_small_holes(mask, area_threshold=int(area_threshold))


def is_metric_crs(crs: CRS) -> bool:
    pycrs = PyCRS.from_user_input(crs)
    if not pycrs.is_projected:
        return False
    axis_info = pycrs.axis_info
    if not axis_info:
        return False
    unit = (axis_info[0].unit_name or "").lower()
    return ("metre" in unit) or ("meter" in unit) or (unit == "m")


def estimate_pixel_area_m2(crs: CRS, transform: Affine, width: int, height: int) -> float:
    """Estimate one-pixel area in square meters for projected or geographic CRS."""
    pycrs = PyCRS.from_user_input(crs)

    if pycrs.is_projected:
        axis_info = pycrs.axis_info
        unit_factor = 1.0
        if axis_info and axis_info[0].unit_conversion_factor:
            unit_factor = float(axis_info[0].unit_conversion_factor)
        px_area_native = abs((transform.a * transform.e) - (transform.b * transform.d))
        px_area_m2 = px_area_native * (unit_factor ** 2)
        if px_area_m2 <= 0:
            raise ValueError("Invalid projected pixel area")
        return px_area_m2

    # Geographic CRS: approximate via geodesic area of center pixel.
    col = width / 2.0
    row = height / 2.0

    corners_px = [
        (col - 0.5, row - 0.5),
        (col + 0.5, row - 0.5),
        (col + 0.5, row + 0.5),
        (col - 0.5, row + 0.5),
    ]
    corners_xy = [transform * p for p in corners_px]

    to_wgs84 = Transformer.from_crs(pycrs, 4326, always_xy=True)
    lonlat = [to_wgs84.transform(x, y) for x, y in corners_xy]

    geod = Geod(ellps="WGS84")
    lons = [p[0] for p in lonlat]
    lats = [p[1] for p in lonlat]
    area, _ = geod.polygon_area_perimeter(lons, lats)
    px_area_m2 = abs(area)
    if px_area_m2 <= 0:
        raise ValueError("Could not estimate geographic pixel area")
    return px_area_m2


def area_m2_to_px(area_m2: float, pixel_area_m2: float) -> int:
    if area_m2 <= 0:
        return 0
    return max(1, int(round(float(area_m2) / float(pixel_area_m2))))


def smooth_probabilities(
    extent_prob: np.ndarray,
    boundary_prob: np.ndarray,
    sigma_px: float,
    valid_mask: np.ndarray,
    output_dtype: str = "float32",
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply light Gaussian smoothing while preserving hard background outside valid mask."""
    out_dtype = np.dtype(output_dtype)
    if out_dtype.kind != "f":
        raise ValueError(f"output_dtype must be floating point, got {output_dtype}")

    sigma = float(sigma_px)
    if sigma <= 0:
        extent_smooth = extent_prob.astype(out_dtype, copy=False)
        boundary_smooth = boundary_prob.astype(out_dtype, copy=False)
    else:
        extent_in = extent_prob.astype(np.float32, copy=False)
        boundary_in = boundary_prob.astype(np.float32, copy=False)
        extent_smooth = ndi.gaussian_filter(extent_in, sigma=sigma, mode="nearest")
        boundary_smooth = ndi.gaussian_filter(boundary_in, sigma=sigma, mode="nearest")
        extent_smooth = extent_smooth.astype(out_dtype, copy=False)
        boundary_smooth = boundary_smooth.astype(out_dtype, copy=False)

    np.clip(extent_smooth, 0.0, 1.0, out=extent_smooth)
    np.clip(boundary_smooth, 0.0, 1.0, out=boundary_smooth)

    inv = valid_mask == 0
    extent_smooth[inv] = 0.0
    boundary_smooth[inv] = 0.0
    return extent_smooth, boundary_smooth


def build_field_mask(
    extent_smooth: np.ndarray,
    valid_mask: np.ndarray,
    extent_thr: float,
    remove_small_objects_px: int,
    fill_holes_px: int,
    opening_px: int = 0,
    closing_px: int = 0,
) -> np.ndarray:
    """Threshold extent and apply conservative binary morphology cleanup."""
    field_mask = extent_smooth >= float(extent_thr)
    field_mask &= valid_mask.astype(bool)

    if remove_small_objects_px > 0:
        field_mask = _remove_small_objects(field_mask, min_size=int(remove_small_objects_px))

    if fill_holes_px > 0:
        field_mask = _remove_small_holes(field_mask, area_threshold=int(fill_holes_px))

    if opening_px > 0:
        field_mask = morphology.binary_opening(field_mask, footprint=morphology.disk(int(opening_px)))

    if closing_px > 0:
        field_mask = morphology.binary_closing(field_mask, footprint=morphology.disk(int(closing_px)))

    field_mask &= valid_mask.astype(bool)
    return field_mask.astype(bool)


def build_boundary_barrier(
    boundary_smooth: np.ndarray,
    valid_mask: np.ndarray,
    boundary_thr: float,
    boundary_dilate_px: int,
) -> np.ndarray:
    """Build binary boundary barrier used to split touching fields."""
    barrier = boundary_smooth >= float(boundary_thr)

    if boundary_dilate_px > 0:
        barrier = morphology.dilation(barrier, footprint=morphology.disk(int(boundary_dilate_px)))

    barrier &= valid_mask.astype(bool)
    return barrier.astype(bool)


def log_thresholds(pixel_area_m2: float, **areas_m2: float) -> None:
    for name, area in areas_m2.items():
        if area <= 0:
            continue
        px = area_m2_to_px(area, pixel_area_m2)
        _LOG.info("%s: %.3f m2 ~= %d px (pixel_area_m2=%.6f)", name, area, px, pixel_area_m2)
