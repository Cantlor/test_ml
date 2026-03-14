from __future__ import annotations

import json
from pathlib import Path

from net_train.infer.predict_aoi import resolve_aoi_path


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def test_resolve_aoi_path_strict_manifest(tmp_path: Path) -> None:
    aoi_tif = tmp_path / "first_raster_aoi.tif"
    aoi_tif.write_bytes(b"")

    manifest = tmp_path / "aoi_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": "1.0",
            "datasets": [
                {
                    "dataset": "first_raster",
                    "status": "clipped",
                    "aoi_raster_path": str(aoi_tif.resolve()),
                }
            ],
        },
    )
    resolved = resolve_aoi_path(manifest, "first_raster")
    assert resolved == aoi_tif.resolve()


def test_resolve_aoi_path_legacy_map_manifest(tmp_path: Path) -> None:
    aoi_tif = tmp_path / "aoi.tif"
    aoi_tif.write_bytes(b"")

    manifest = tmp_path / "aoi_rasters_manifest.json"
    _write_json(manifest, {"first_raster": str(aoi_tif.resolve())})

    resolved = resolve_aoi_path(manifest, "first_raster")
    assert resolved == aoi_tif.resolve()


def test_resolve_aoi_path_legacy_results_manifest(tmp_path: Path) -> None:
    aoi_tif = tmp_path / "aoi_legacy.tif"
    aoi_tif.write_bytes(b"")

    manifest = tmp_path / "aoi_rasters_manifest.json"
    _write_json(
        manifest,
        {
            "results": [
                {
                    "dataset": "first_raster",
                    "out_raster": str(aoi_tif.resolve()),
                }
            ]
        },
    )

    resolved = resolve_aoi_path(manifest, "first_raster")
    assert resolved == aoi_tif.resolve()
