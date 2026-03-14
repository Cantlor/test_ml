from __future__ import annotations

import json
from pathlib import Path

import pytest

from prep.manifests import (
    AoiDatasetResult,
    AoiManifest,
    CheckInputsDatasetResult,
    CheckInputsManifest,
    PatchesDatasetResult,
    PatchesManifest,
    SplitManifest,
)


def test_check_inputs_manifest_roundtrip(tmp_path: Path):
    m = CheckInputsManifest.new(
        config_path=tmp_path / "prep_config.yaml",
        work_dir=tmp_path / "work",
        data_check_json_path=tmp_path / "data_check.json",
        datasets=[
            CheckInputsDatasetResult(
                dataset="ds1",
                root=str(tmp_path / "data" / "ds1"),
                raw_raster_path=str(tmp_path / "data" / "ds1" / "raw.tif"),
                raw_vector_path=str(tmp_path / "data" / "ds1" / "raw.shp"),
                vector_layer=None,
                vector_id_field=None,
                qa_raster_path=str(tmp_path / "data" / "ds1" / "raw.tif"),
                qa_raster_source="raw",
                prepared_vector_path=str(tmp_path / "work" / "ds1_prepared.gpkg"),
                prepared_vector_layer="fields_prepared",
                qa_ok=True,
            )
        ],
        deferred_config_keys=["raster_preprocess.*"],
    )
    path = m.save(tmp_path / "work" / "check_inputs_manifest.json")
    m2 = CheckInputsManifest.load(path)
    assert m2.schema_version == m.schema_version
    assert len(m2.datasets) == 1
    assert m2.datasets[0].prepared_vector_layer == "fields_prepared"


def test_manifest_missing_required_key_is_strict(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    with pytest.raises(RuntimeError):
        _ = CheckInputsManifest.load(p)


def test_patches_manifest_roundtrip(tmp_path: Path):
    m = PatchesManifest.new(
        config_path=tmp_path / "prep_config.yaml",
        work_dir=tmp_path / "work",
        patches_all_root=tmp_path / "work" / "patches_all",
        datasets=[
            PatchesDatasetResult(
                dataset="ds1",
                raster_path=str(tmp_path / "work" / "aoi.tif"),
                raster_source="aoi",
                vector_path=str(tmp_path / "work" / "vec.gpkg"),
                vector_layer="fields_prepared",
                vector_id_field=None,
                dataset_manifest_path=str(tmp_path / "work" / "patches_all" / "ds1" / "manifest.json"),
                cleaned_vector_raster_crs_path=None,
                output_dataset_dir=str(tmp_path / "work" / "patches_all" / "ds1"),
                status="ok",
            )
        ],
    )
    p = m.save(tmp_path / "work" / "patches_manifest.json")
    loaded = PatchesManifest.load(p)
    assert loaded.datasets[0].dataset == "ds1"
    assert loaded.datasets[0].raster_source == "aoi"


def test_split_manifest_roundtrip(tmp_path: Path):
    m = SplitManifest.new(
        config_path=tmp_path / "prep_config.yaml",
        patches_manifest_path=tmp_path / "work" / "patches_manifest.json",
        mode="append",
        seed=123,
        ratios={"train": 0.8, "validation": 0.1, "test": 0.1},
        split_roots={"train": tmp_path / "prep_data" / "train", "validation": tmp_path / "prep_data" / "validation", "test": tmp_path / "prep_data" / "test"},
        export_folders={
            "img": "img",
            "extent": "extent",
            "extent_ig": "extent_ig",
            "boundary_raw": "boundary_raw",
            "boundary_bwbl": "boundary_bwbl",
            "valid": "valid",
            "meta": "meta",
        },
        assign_info={"new_total": 10},
        copied_new_counts={"train": 8, "validation": 1, "test": 1},
        final_meta_counts={"train": 8, "validation": 1, "test": 1},
        notes={"extent_source": "extent_ig"},
    )
    p = m.save(tmp_path / "prep_data" / "split_manifest.json")
    loaded = SplitManifest.load(p)
    assert loaded.mode == "append"
    assert loaded.notes["extent_source"] == "extent_ig"
