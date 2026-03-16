from __future__ import annotations

import json
from pathlib import Path

from module_postprocess_vectorize.postprocess.inputs import (
    discover_prediction_samples,
    resolve_prediction_sample_from_manifest,
    resolve_prediction_sample_from_run,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_resolve_prediction_sample_uses_current_predict_manifest_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_1"
    pred_dir = run_dir / "pred" / "first_raster"
    extent_path = _touch(pred_dir / "extent_prob.tif")
    boundary_path = _touch(pred_dir / "boundary_prob.tif")
    aoi_path = _touch(tmp_path / "aoi" / "first_raster_aoi.tif")
    config_path = run_dir / "config_resolved.yaml"
    config_path.write_text(
        "\n".join(
            [
                "dataset:",
                "  inputs:",
                "    nodata_value: 65536",
                "    nodata_rule: control-band",
                "    control_band_1based: 1",
            ]
        ),
        encoding="utf-8",
    )

    manifest_path = pred_dir / "predict_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "extent_prob": str(extent_path),
                "boundary_prob": str(boundary_path),
                "aoi_raster": str(aoi_path),
                "config_used": str(config_path),
            }
        ),
        encoding="utf-8",
    )

    sample = resolve_prediction_sample_from_manifest(manifest_path)
    assert sample.sample_id == "first_raster"
    assert sample.extent_prob_path == extent_path.resolve()
    assert sample.boundary_prob_path == boundary_path.resolve()
    assert sample.footprint_path == aoi_path.resolve()
    assert sample.config_used_path == config_path.resolve()
    assert sample.valid_nodata_value == 65536.0
    assert sample.valid_nodata_rule == "control-band"
    assert sample.valid_control_band_1based == 1

    discovered = discover_prediction_samples(run_dir / "pred")
    assert [item.sample_id for item in discovered] == ["first_raster"]
    assert discovered[0].footprint_path == aoi_path.resolve()

    from_run = resolve_prediction_sample_from_run(run_dir, dataset_key="first_raster")
    assert from_run.predict_manifest_path == manifest_path.resolve()
    assert from_run.config_used_path == config_path.resolve()
