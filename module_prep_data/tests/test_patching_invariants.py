from __future__ import annotations

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import LineString, Polygon, box

from prep.patching.labels import (
    apply_nodata_ignore_policy,
    boundary_raw_from_extent_gradient,
    extent_and_boundaries_for_window,
)
from prep.patching.nodata import valid_mask_from_chip


def test_valid_mask_control_band_rule():
    chip = np.array(
        [
            [[65536, 1], [1, 65536]],
            [[10, 10], [10, 10]],
        ],
        dtype=np.int32,
    )
    valid = valid_mask_from_chip(
        chip=chip,
        nodata_value=65536,
        nodata_rule="control-band",
        control_band_1based=1,
    )
    expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(valid, expected)


def test_valid_mask_all_bands_rule():
    chip = np.array(
        [
            [[65536, 1], [65536, 65536]],
            [[65536, 1], [2, 65536]],
        ],
        dtype=np.int32,
    )
    valid = valid_mask_from_chip(
        chip=chip,
        nodata_value=65536,
        nodata_rule="all-bands",
        control_band_1based=1,
    )
    expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(valid, expected)


def test_nodata_ignore_policy_semantics():
    extent_ig = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    bwbl = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    valid = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    e2, b2 = apply_nodata_ignore_policy(
        extent_ig=extent_ig,
        bwbl=bwbl,
        valid_u8=valid,
        extent_ignore_value=255,
        bwbl_ignore_value=2,
    )

    assert int(e2[0, 1]) == 255
    assert int(e2[1, 1]) == 255
    assert int(b2[0, 1]) == 2
    assert int(b2[1, 1]) == 2


def _run_extent_and_boundaries(
    gdf: gpd.GeoDataFrame,
    *,
    include_holes: bool,
    ignore_enabled: bool,
):
    transform = from_origin(0.0, 32.0, 1.0, 1.0)
    with MemoryFile() as mem:
        with mem.open(
            driver="GTiff",
            width=32,
            height=32,
            count=1,
            dtype="uint8",
            transform=transform,
            crs="EPSG:3857",
        ) as ds:
            ds.write(np.zeros((1, 32, 32), dtype=np.uint8))
            try:
                sindex = gdf.sindex
            except Exception:
                sindex = None
            out = extent_and_boundaries_for_window(
                ds=ds,
                gdf=gdf,
                sindex=sindex,
                win=Window(0, 0, 32, 32),
                include_holes=include_holes,
                ignore_enabled=ignore_enabled,
                ignore_value=255,
                ignore_apply_to_extent=True,
                ignore_radius_px=1,
                pad_px=0,
                bwbl_buffer_px=2,
                bwbl_background_value=0,
                bwbl_skeleton_value=1,
                bwbl_buffer_value=2,
            )
    return (*out, transform)


def _line_mask(transform, line) -> np.ndarray:
    return rasterize(
        [(line, 1)],
        out_shape=(32, 32),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )


def test_boundary_raw_keeps_shared_internal_boundaries():
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(2, 2, 16, 20), box(16, 2, 30, 20)],
        crs="EPSG:3857",
    )
    extent, _, boundary_raw, _, _, transform = _run_extent_and_boundaries(
        gdf,
        include_holes=True,
        ignore_enabled=False,
    )

    boundary_old = boundary_raw_from_extent_gradient(extent)
    inner_line = _line_mask(transform, LineString([(16, 2), (16, 20)]))
    outer_line = _line_mask(transform, LineString([(2, 2), (2, 20)]))

    inner_hit_new = int(((boundary_raw > 0) & (inner_line > 0)).sum())
    inner_hit_old = int(((boundary_old > 0) & (inner_line > 0)).sum())
    outer_hit_new = int(((boundary_raw > 0) & (outer_line > 0)).sum())

    assert inner_hit_new > 0
    assert outer_hit_new > 0
    assert inner_hit_new >= inner_hit_old + 8


def test_include_holes_flag_controls_hole_boundaries():
    geom = Polygon(
        shell=[(2, 2), (30, 2), (30, 30), (2, 30), (2, 2)],
        holes=[[(10, 10), (22, 10), (22, 22), (10, 22), (10, 10)]],
    )
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:3857")

    _, _, braw_keep, _, _, transform = _run_extent_and_boundaries(
        gdf,
        include_holes=True,
        ignore_enabled=False,
    )
    _, _, braw_drop, _, _, _ = _run_extent_and_boundaries(
        gdf,
        include_holes=False,
        ignore_enabled=False,
    )

    hole_edge = _line_mask(transform, LineString([(10, 10), (22, 10)]))
    keep_hits = int(((braw_keep > 0) & (hole_edge > 0)).sum())
    drop_hits = int(((braw_drop > 0) & (hole_edge > 0)).sum())

    assert keep_hits > 0
    assert drop_hits == 0


def test_boundary_bwbl_and_extent_contracts_remain_compatible():
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[box(2, 2, 16, 20), box(16, 2, 30, 20)],
        crs="EPSG:3857",
    )
    extent, extent_ig, _, bwbl, _, transform = _run_extent_and_boundaries(
        gdf,
        include_holes=True,
        ignore_enabled=True,
    )

    assert set(np.unique(extent)).issubset({0, 1})
    assert set(np.unique(extent_ig)).issubset({0, 1, 255})
    assert set(np.unique(bwbl)).issubset({0, 1, 2})

    inner_line = _line_mask(transform, LineString([(16, 2), (16, 20)]))
    inner_signal = int((((bwbl == 1) | (bwbl == 2)) & (inner_line > 0)).sum())
    assert inner_signal > 0
