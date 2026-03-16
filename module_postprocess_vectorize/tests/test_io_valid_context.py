from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from module_postprocess_vectorize.postprocess.io import RasterMeta, load_inputs, save_raster


def _write_single_band_raster(path: Path, array: np.ndarray, *, dtype: str, nodata: float | None = None) -> Path:
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
        nodata=nodata,
    ) as ds:
        ds.write(array.astype(dtype), 1)
    return path


def test_load_inputs_builds_valid_mask_from_footprint_nodata(tmp_path: Path) -> None:
    extent_prob = np.array([[0.8, 0.4], [0.6, 0.2]], dtype=np.float32)
    boundary_prob = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    # Top-left pixel is a valid zero; bottom-left pixel is invalid nodata.
    footprint = np.array([[0, 10], [65535, 20]], dtype=np.uint16)

    extent_path = _write_single_band_raster(tmp_path / "extent_prob.tif", extent_prob, dtype="float32")
    boundary_path = _write_single_band_raster(tmp_path / "boundary_prob.tif", boundary_prob, dtype="float32")
    footprint_path = _write_single_band_raster(tmp_path / "aoi_raster.tif", footprint, dtype="uint16", nodata=65535)

    bundle = load_inputs(
        extent_prob_path=extent_path,
        boundary_prob_path=boundary_path,
        footprint_path=footprint_path,
    )

    expected_valid = np.array([[1, 1], [0, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(bundle.valid_mask, expected_valid)
    assert bundle.valid_source == "footprint_nodata"
    assert bundle.valid_context["nodata_value"] == 65535.0
    # Invalid pixels are hard-zeroed in the probability rasters.
    assert float(bundle.extent_prob[1, 0]) == 0.0
    assert float(bundle.boundary_prob[1, 0]) == 0.0


def test_save_raster_normalizes_tile_block_sizes_for_small_outputs(tmp_path: Path) -> None:
    transform = from_origin(500000.0, 4500000.0, 3.0, 3.0)
    meta = RasterMeta(
        width=32,
        height=32,
        transform=transform,
        crs=CRS.from_epsg(32642),
        profile={
            "driver": "GTiff",
            "dtype": "float32",
            "count": 1,
            "width": 32,
            "height": 32,
            "transform": transform,
            "crs": CRS.from_epsg(32642),
            "tiled": True,
            "blockxsize": 500,
            "blockysize": 500,
        },
    )

    out_path = save_raster(
        tmp_path / "small.tif",
        np.ones((32, 32), dtype=np.float32),
        meta,
        dtype="float32",
        nodata=0.0,
    )

    with rasterio.open(out_path) as ds:
        block_h, block_w = ds.block_shapes[0]
        assert block_h % 16 == 0
        assert block_w % 16 == 0
