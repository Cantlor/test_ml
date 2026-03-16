from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from module_postprocess_vectorize.postprocess.pipeline import run_postprocess_pipeline


def _write_single_band(path: Path, array: np.ndarray, *, dtype: str = "float32") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        count=1,
        dtype=dtype,
        crs="EPSG:32642",
        transform=from_origin(500000.0, 4500000.0, 3.0, 3.0),
        nodata=0,
    ) as ds:
        ds.write(array.astype(dtype), 1)
    return path


def _base_config() -> dict:
    return {
        "extent_thr": 0.50,
        "boundary_thr": 0.50,
        "gaussian_sigma_px": 1.0,
        "boundary_dilate_px": 1,
        "min_area_m2": 5.0,
        "fill_holes_max_area_m2": 5.0,
        "small_region_max_area_m2": 5.0,
        "remove_small_objects_m2": 5.0,
        "seed_min_distance_px": 2,
        "seed_hmax": 1.0,
        "marker_erode_px": 0,
        "use_watershed": True,
        "boundary_weight": 2.0,
        "sobel_weight": 0.1,
        "remove_holes": False,
        "simplify_m": 0.0,
        "clip_to_valid": False,
        "save_intermediates": False,
        "memory": {
            "prob_dtype": "float32",
            "smooth_dtype": "auto",
            "max_pixels_for_watershed": 50_000_000,
            "max_pixels_for_gaussian": 50_000_000,
            "clean_labels_mode": "auto",
        },
    }


def test_pipeline_early_exit_when_all_pixels_invalid(tmp_path: Path) -> None:
    extent = np.full((4, 4), 0.9, dtype=np.float32)
    boundary = np.full((4, 4), 0.6, dtype=np.float32)
    valid = np.zeros((4, 4), dtype=np.uint8)

    extent_path = _write_single_band(tmp_path / "extent_prob.tif", extent, dtype="float32")
    boundary_path = _write_single_band(tmp_path / "boundary_prob.tif", boundary, dtype="float32")
    valid_path = _write_single_band(tmp_path / "valid_mask.tif", valid, dtype="uint8")

    out = run_postprocess_pipeline(
        extent_prob_path=extent_path,
        boundary_prob_path=boundary_path,
        valid_mask_path=valid_path,
        output_dir=tmp_path / "out_all_invalid",
        config=_base_config(),
        save_outputs=False,
    )

    assert out["fields_pred"].empty
    assert out["labels_stats"]["num_labels"] == 0
    assert out["memory_runtime"]["early_exit_reason"] == "all_invalid"


def test_pipeline_early_exit_when_extent_never_reaches_threshold(tmp_path: Path) -> None:
    extent = np.full((4, 4), 0.2, dtype=np.float32)
    boundary = np.full((4, 4), 0.1, dtype=np.float32)
    valid = np.ones((4, 4), dtype=np.uint8)

    extent_path = _write_single_band(tmp_path / "extent_prob.tif", extent, dtype="float32")
    boundary_path = _write_single_band(tmp_path / "boundary_prob.tif", boundary, dtype="float32")
    valid_path = _write_single_band(tmp_path / "valid_mask.tif", valid, dtype="uint8")

    out = run_postprocess_pipeline(
        extent_prob_path=extent_path,
        boundary_prob_path=boundary_path,
        valid_mask_path=valid_path,
        output_dir=tmp_path / "out_empty",
        config=_base_config(),
        save_outputs=False,
    )

    assert out["fields_pred"].empty
    assert out["labels_stats"]["num_labels"] == 0
    assert out["memory_runtime"]["early_exit_reason"] == "no_extent_above_threshold"
