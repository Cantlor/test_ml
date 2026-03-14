from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from prep.manifests import PatchesDatasetResult, PatchesManifest, SplitManifest
from prep.stages.split_dataset import run


def _write_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        count = 1
        h, w = arr.shape
    else:
        count, h, w = arr.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=int(h),
        width=int(w),
        count=int(count),
        dtype=str(arr.dtype),
        crs="EPSG:4326",
        transform=from_origin(0, 0, 1, 1),
    ) as ds:
        if arr.ndim == 2:
            ds.write(arr, 1)
        else:
            ds.write(arr)


def test_split_stage_uses_configured_export_structure_and_folders(tmp_path: Path):
    cfg_path = tmp_path / "prep_config.yaml"
    cfg_path.write_text(
        """
version: 1
paths:
  work_dir: ./work
  prep_data_dir: ./prep_state
datasets: []
split:
  ratios:
    train: 1.0
    validation: 0.0
    test: 0.0
  unit: by_field
  seed: 7
export:
  structure:
    train_dir: ./exports/TR
    validation_dir: ./exports/VA
    test_dir: ./exports/TE
  folders:
    img: image
    extent: ext
    extent_ig: ext_ig
    boundary_raw: braw
    boundary_bwbl: bw
    valid: v
    meta: m
""".strip()
        + "\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    ds_dir = work_dir / "patches_all" / "ds1"
    patch_id = "ds1_000000"
    meta = {
        "dataset": "ds1",
        "patch_id": patch_id,
        "inside_mode": "center",
        "feat_index": 0,
        "field_id": "ds1::f0",
    }

    _write_tif(ds_dir / "img" / f"img_{patch_id}.tif", np.ones((1, 2, 2), dtype=np.uint16))
    _write_tif(ds_dir / "extent_ig" / f"extent_ig_{patch_id}.tif", np.full((2, 2), 255, dtype=np.uint8))
    _write_tif(ds_dir / "boundary_raw" / f"boundary_raw_{patch_id}.tif", np.ones((2, 2), dtype=np.uint8))
    _write_tif(ds_dir / "boundary_bwbl" / f"bwbl_{patch_id}.tif", np.full((2, 2), 2, dtype=np.uint8))
    _write_tif(ds_dir / "valid" / f"valid_{patch_id}.tif", np.zeros((2, 2), dtype=np.uint8))
    (ds_dir / "meta").mkdir(parents=True, exist_ok=True)
    (ds_dir / "meta" / f"meta_{patch_id}.json").write_text(json.dumps(meta), encoding="utf-8")

    ds_manifest_path = ds_dir / "manifest.json"
    ds_manifest_path.write_text(json.dumps({"summary": {"dataset": "ds1"}, "patches": [meta]}), encoding="utf-8")

    top = PatchesManifest.new(
        config_path=cfg_path,
        work_dir=work_dir,
        patches_all_root=work_dir / "patches_all",
        datasets=[
            PatchesDatasetResult(
                dataset="ds1",
                raster_path=str(tmp_path / "raw.tif"),
                raster_source="raw",
                vector_path=str(tmp_path / "vec.shp"),
                vector_layer=None,
                vector_id_field=None,
                dataset_manifest_path=str(ds_manifest_path),
                cleaned_vector_raster_crs_path=None,
                output_dataset_dir=str(ds_dir),
                status="ok",
            )
        ],
    )
    top_path = top.save(work_dir / "patches_manifest.json")

    rc = run(config_path=cfg_path, patches_manifest_override=top_path, overwrite=True)
    assert rc == 0

    train_root = tmp_path / "exports" / "TR"
    assert (train_root / "image" / f"img_{patch_id}.tif").exists()
    assert (train_root / "ext" / f"extent_{patch_id}.tif").exists()
    assert (train_root / "ext_ig" / f"extent_ig_{patch_id}.tif").exists()
    assert (train_root / "bw" / f"bwbl_{patch_id}.tif").exists()
    assert (train_root / "v" / f"valid_{patch_id}.tif").exists()
    assert (train_root / "m" / f"meta_{patch_id}.json").exists()

    with rasterio.open(ds_dir / "extent_ig" / f"extent_ig_{patch_id}.tif") as src, rasterio.open(
        train_root / "ext" / f"extent_{patch_id}.tif"
    ) as dst:
        np.testing.assert_array_equal(src.read(1), dst.read(1))

    rc2 = run(config_path=cfg_path, patches_manifest_override=top_path, overwrite=False)
    assert rc2 == 0

    split_manifest_path = tmp_path / "prep_state" / "split_manifest.json"
    sm = SplitManifest.load(split_manifest_path)
    assert sm.notes["extent_source"] == "extent_ig (0/1/255)"
    assert sm.copied_new_counts["train"] == 0
