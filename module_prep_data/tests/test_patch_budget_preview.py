from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from prep.artifacts import PREPARED_VECTOR_LAYER
from prep.manifests import CheckInputsDatasetResult, CheckInputsManifest
from prep.stages.make_patches import run


def _write_raster(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ones((1, 64, 64), dtype=np.uint16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype=str(arr.dtype),
        crs="EPSG:3857",
        transform=from_origin(0, 64, 1, 1),
    ) as ds:
        ds.write(arr)


def _write_vector(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(8, 8, 40, 40)], crs="EPSG:3857")
    gdf.to_file(path, driver="GPKG", layer=PREPARED_VECTOR_LAYER)


def _setup_case(tmp_path: Path, *, patch_budget_mode: str, target_patches_per_dataset: int) -> tuple[Path, Path]:
    cfg_path = tmp_path / "prep_config.yaml"
    cfg_path.write_text(
        f"""
version: 1
paths:
  work_dir: ./work
datasets:
  - name: ds1
    root: ./input/ds1
    raster:
      glob: ["*.tif"]
      require_single_match: true
    vector:
      glob: ["*.gpkg"]
      require_single_match: true
aoi_clip:
  enabled: false
patching:
  patch_size_px: 64
  target_patches_per_dataset: {int(target_patches_per_dataset)}
  patch_budget:
    mode: {patch_budget_mode}
    min_valid_patches_to_keep: 1
performance:
  progress: false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    work_dir = tmp_path / "work"
    raster_path = tmp_path / "input" / "ds1" / "raw.tif"
    vector_path = work_dir / "ds1_vector_prepared.gpkg"
    _write_raster(raster_path)
    _write_vector(vector_path)

    check_manifest = CheckInputsManifest.new(
        config_path=cfg_path,
        work_dir=work_dir,
        data_check_json_path=tmp_path / "data_check.json",
        datasets=[
            CheckInputsDatasetResult(
                dataset="ds1",
                root=str(tmp_path / "input" / "ds1"),
                raw_raster_path=str(raster_path),
                raw_vector_path=str(vector_path),
                vector_layer=None,
                vector_id_field=None,
                qa_raster_path=str(raster_path),
                qa_raster_source="raw",
                prepared_vector_path=str(vector_path),
                prepared_vector_layer=PREPARED_VECTOR_LAYER,
                qa_ok=True,
            )
        ],
    )
    check_manifest.save(work_dir / "check_inputs_manifest.json")
    return cfg_path, work_dir


def test_patch_budget_preview_auto_creates_report_only(tmp_path: Path):
    cfg_path, work_dir = _setup_case(tmp_path, patch_budget_mode="auto", target_patches_per_dataset=40)

    rc = run(config_path=cfg_path, dry_run_budget=True)
    assert rc == 0

    report_path = work_dir / "patch_budget_report.json"
    assert report_path.exists()
    obj = json.loads(report_path.read_text(encoding="utf-8"))

    assert obj["dry_run_budget"] is True
    assert obj["patch_budget_mode"] == "auto"
    assert obj["summary"]["datasets_checked"] == 1
    assert obj["summary"]["datasets_skipped_effective"] in {0, 1}
    assert len(obj["datasets"]) == 1
    ds = obj["datasets"][0]
    assert ds["dataset"] == "ds1"
    assert "auto_budget_info" in ds
    assert "estimated_total_capacity" in ds["auto_budget_info"]

    assert not (work_dir / "patches_manifest.json").exists()
    assert not (work_dir / "patches_all").exists()


def test_patch_budget_preview_fixed_shows_effective_fixed_target(tmp_path: Path):
    cfg_path, work_dir = _setup_case(tmp_path, patch_budget_mode="fixed", target_patches_per_dataset=37)

    rc = run(config_path=cfg_path, dry_run_budget=True)
    assert rc == 0

    obj = json.loads((work_dir / "patch_budget_report.json").read_text(encoding="utf-8"))
    assert obj["patch_budget_mode"] == "fixed"
    ds = obj["datasets"][0]
    assert ds["effective_targets"]["target_total"] == 37
    assert "estimated_total_capacity" in ds["auto_budget_info"]
