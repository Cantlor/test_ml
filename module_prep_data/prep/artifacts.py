from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import Config, DatasetSpec
from .manifests import (
    AoiDatasetResult,
    AoiManifest,
    CheckInputsDatasetResult,
    CheckInputsManifest,
    PatchesDatasetResult,
    PatchesManifest,
    load_manifest,
)
from .utils import find_single_by_globs


CHECK_INPUTS_MANIFEST_NAME = "check_inputs_manifest.json"
AOI_MANIFEST_NAME = "aoi_manifest.json"
PATCHES_MANIFEST_NAME = "patches_manifest.json"
SPLIT_MANIFEST_NAME = "split_manifest.json"
PREPARED_VECTOR_LAYER = "fields_prepared"


def get_work_dir(cfg: Config) -> Path:
    return Path(cfg.paths.get("work_dir", (cfg.project_root / "../output_data/module_prep_data_work").resolve())).resolve()


def get_prep_data_root(cfg: Config) -> Path:
    return Path(cfg.paths.get("prep_data_dir", (cfg.project_root / "../prep_data").resolve())).resolve()


def check_inputs_manifest_path(cfg: Config) -> Path:
    return get_work_dir(cfg) / CHECK_INPUTS_MANIFEST_NAME


def aoi_manifest_path(cfg: Config) -> Path:
    return get_work_dir(cfg) / AOI_MANIFEST_NAME


def patches_manifest_path(cfg: Config) -> Path:
    return get_work_dir(cfg) / PATCHES_MANIFEST_NAME


def split_manifest_path(cfg: Config) -> Path:
    return get_prep_data_root(cfg) / SPLIT_MANIFEST_NAME


def resolve_raw_inputs(ds: DatasetSpec) -> Tuple[Path, Path]:
    raster_path, raster_matches = find_single_by_globs(ds.root, ds.raster_glob, ds.raster_require_single)
    if raster_path is None:
        raise RuntimeError(f"{ds.name}: expected single raster by {ds.raster_glob}, got {len(raster_matches)}")

    vector_path, vector_matches = find_single_by_globs(ds.root, ds.vector_glob, ds.vector_require_single)
    if vector_path is None:
        raise RuntimeError(f"{ds.name}: expected single vector by {ds.vector_glob}, got {len(vector_matches)}")
    return raster_path, vector_path


def load_check_inputs_manifest_required(cfg: Config) -> CheckInputsManifest:
    p = check_inputs_manifest_path(cfg)
    return load_manifest(p, CheckInputsManifest)


def load_aoi_manifest_required(cfg: Config) -> AoiManifest:
    p = aoi_manifest_path(cfg)
    return load_manifest(p, AoiManifest)


def load_patches_manifest_required_from_path(path: str | Path) -> PatchesManifest:
    return load_manifest(path, PatchesManifest)


def load_patches_manifest_required(cfg: Config) -> PatchesManifest:
    p = patches_manifest_path(cfg)
    return load_manifest(p, PatchesManifest)


def check_dataset_entry(m: CheckInputsManifest, ds_name: str) -> CheckInputsDatasetResult:
    for item in m.datasets:
        if item.dataset == ds_name:
            return item
    raise RuntimeError(f"check_inputs manifest has no dataset entry for '{ds_name}'")


def aoi_dataset_entry(m: AoiManifest, ds_name: str) -> AoiDatasetResult:
    for item in m.datasets:
        if item.dataset == ds_name:
            return item
    raise RuntimeError(f"aoi manifest has no dataset entry for '{ds_name}'")


def patches_dataset_entry(m: PatchesManifest, ds_name: str) -> PatchesDatasetResult:
    for item in m.datasets:
        if item.dataset == ds_name:
            return item
    raise RuntimeError(f"patches manifest has no dataset entry for '{ds_name}'")


def resolve_patch_inputs_for_dataset(
    cfg: Config,
    ds: DatasetSpec,
    check_manifest: CheckInputsManifest,
    aoi_manifest: Optional[AoiManifest],
) -> Tuple[Path, str, Path, Optional[str]]:
    c = check_dataset_entry(check_manifest, ds.name)

    if not c.prepared_vector_path:
        raise RuntimeError(
            f"{ds.name}: prepared_vector_path missing in {CHECK_INPUTS_MANIFEST_NAME}. "
            "Run scripts/01_check_inputs.py first."
        )
    vector_path = Path(c.prepared_vector_path).resolve()
    if not vector_path.exists():
        raise RuntimeError(f"{ds.name}: prepared vector not found: {vector_path}")
    vector_layer = c.prepared_vector_layer or PREPARED_VECTOR_LAYER

    if cfg.aoi_clip.enabled:
        if aoi_manifest is None:
            raise RuntimeError(
                f"{ds.name}: AOI is enabled, but {AOI_MANIFEST_NAME} was not provided. "
                "Run scripts/02_clip_to_aoi.py first."
            )
        a = aoi_dataset_entry(aoi_manifest, ds.name)
        if a.status != "clipped" or not a.aoi_raster_path:
            raise RuntimeError(f"{ds.name}: AOI entry is not clipped in {AOI_MANIFEST_NAME} (status={a.status})")
        raster_path = Path(a.aoi_raster_path).resolve()
        if not raster_path.exists():
            raise RuntimeError(f"{ds.name}: AOI raster not found: {raster_path}")
        return raster_path, "aoi", vector_path, vector_layer

    raster_path = Path(c.raw_raster_path).resolve()
    if not raster_path.exists():
        raise RuntimeError(f"{ds.name}: raw raster not found: {raster_path}")
    return raster_path, "raw", vector_path, vector_layer


def export_split_roots_from_cfg(cfg: Config) -> Dict[str, Path]:
    return {
        "train": Path(cfg.export.structure.train_dir).resolve()
        if Path(cfg.export.structure.train_dir).is_absolute()
        else (cfg.project_root / cfg.export.structure.train_dir).resolve(),
        "validation": Path(cfg.export.structure.validation_dir).resolve()
        if Path(cfg.export.structure.validation_dir).is_absolute()
        else (cfg.project_root / cfg.export.structure.validation_dir).resolve(),
        "test": Path(cfg.export.structure.test_dir).resolve()
        if Path(cfg.export.structure.test_dir).is_absolute()
        else (cfg.project_root / cfg.export.structure.test_dir).resolve(),
    }


def export_folders_from_cfg(cfg: Config) -> Dict[str, str]:
    f = cfg.export.folders
    return {
        "img": str(f.img),
        "extent": str(f.extent),
        "extent_ig": str(f.extent_ig),
        "boundary_raw": str(f.boundary_raw),
        "boundary_bwbl": str(f.boundary_bwbl),
        "valid": str(f.valid),
        "meta": str(f.meta),
    }
