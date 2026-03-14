from __future__ import annotations

from pathlib import Path

import pytest

from prep.artifacts import resolve_patch_inputs_for_dataset
from prep.config import load_config
from prep.manifests import (
    AoiDatasetResult,
    AoiManifest,
    CheckInputsDatasetResult,
    CheckInputsManifest,
)


def _write_cfg(path: Path, aoi_enabled: bool) -> Path:
    txt = f"""
version: 1
paths:
  work_dir: ./work
datasets:
  - name: ds1
    root: ./data/ds1
    raster:
      glob: ["*.tif"]
      require_single_match: true
    vector:
      glob: ["*.shp"]
      require_single_match: true
aoi_clip:
  enabled: {"true" if aoi_enabled else "false"}
  out_dir: ./work/aoi_rasters
"""
    path.write_text(txt.strip() + "\n", encoding="utf-8")
    return path


def test_resolve_patch_inputs_prefers_aoi_in_strict_manifest_mode(tmp_path: Path):
    cfg_path = _write_cfg(tmp_path / "prep_config.yaml", aoi_enabled=True)
    cfg = load_config(cfg_path)
    ds = cfg.datasets[0]

    raw_raster = tmp_path / "data" / "ds1" / "raw.tif"
    raw_vector = tmp_path / "data" / "ds1" / "raw.shp"
    prepared = tmp_path / "work" / "ds1_vector_prepared.gpkg"
    aoi = tmp_path / "work" / "aoi_rasters" / "ds1_aoi.tif"
    for p in [raw_raster, raw_vector, prepared, aoi]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")

    check_m = CheckInputsManifest.new(
        config_path=cfg_path,
        work_dir=tmp_path / "work",
        data_check_json_path=tmp_path / "out" / "data_check.json",
        datasets=[
            CheckInputsDatasetResult(
                dataset="ds1",
                root=str(tmp_path / "data" / "ds1"),
                raw_raster_path=str(raw_raster),
                raw_vector_path=str(raw_vector),
                vector_layer=None,
                vector_id_field=None,
                qa_raster_path=str(raw_raster),
                qa_raster_source="raw",
                prepared_vector_path=str(prepared),
                prepared_vector_layer="fields_prepared",
                qa_ok=True,
            )
        ],
    )
    aoi_m = AoiManifest.new(
        config_path=cfg_path,
        work_dir=tmp_path / "work",
        out_dir=tmp_path / "work" / "aoi_rasters",
        enabled=True,
        datasets=[
            AoiDatasetResult(
                dataset="ds1",
                source_raster_path=str(raw_raster),
                vector_path=str(raw_vector),
                vector_layer=None,
                aoi_raster_path=str(aoi),
                mode="bbox",
                buffer_m=0.0,
                mask_outside=False,
                wrote_mask_outside=False,
                status="clipped",
            )
        ],
    )

    raster_path, raster_source, vector_path, vector_layer = resolve_patch_inputs_for_dataset(
        cfg=cfg,
        ds=ds,
        check_manifest=check_m,
        aoi_manifest=aoi_m,
    )
    assert raster_source == "aoi"
    assert raster_path == aoi.resolve()
    assert vector_path == prepared.resolve()
    assert vector_layer == "fields_prepared"


def test_resolve_patch_inputs_requires_aoi_manifest_when_enabled(tmp_path: Path):
    cfg_path = _write_cfg(tmp_path / "prep_config.yaml", aoi_enabled=True)
    cfg = load_config(cfg_path)
    ds = cfg.datasets[0]

    raw_raster = tmp_path / "data" / "ds1" / "raw.tif"
    raw_vector = tmp_path / "data" / "ds1" / "raw.shp"
    prepared = tmp_path / "work" / "ds1_vector_prepared.gpkg"
    for p in [raw_raster, raw_vector, prepared]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")

    check_m = CheckInputsManifest.new(
        config_path=cfg_path,
        work_dir=tmp_path / "work",
        data_check_json_path=tmp_path / "out" / "data_check.json",
        datasets=[
            CheckInputsDatasetResult(
                dataset="ds1",
                root=str(tmp_path / "data" / "ds1"),
                raw_raster_path=str(raw_raster),
                raw_vector_path=str(raw_vector),
                vector_layer=None,
                vector_id_field=None,
                qa_raster_path=str(raw_raster),
                qa_raster_source="raw",
                prepared_vector_path=str(prepared),
                prepared_vector_layer="fields_prepared",
                qa_ok=True,
            )
        ],
    )

    with pytest.raises(RuntimeError):
        _ = resolve_patch_inputs_for_dataset(
            cfg=cfg,
            ds=ds,
            check_manifest=check_m,
            aoi_manifest=None,
        )
