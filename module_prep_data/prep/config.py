#config.py

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Tuple

import yaml


def _abs_from(base: Path, p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (base / pp).resolve()


# ----------------------------
# Dataset spec
# ----------------------------
@dataclass
class DatasetSpec:
    name: str
    root: Path
    raster_glob: List[str]
    raster_require_single: bool
    vector_glob: List[str]
    vector_require_single: bool
    vector_layer: Optional[str] = None
    vector_id_field: Optional[str] = None


# ----------------------------
# Typed sub-configs
# ----------------------------
NoDataRule = Literal["control-band", "all-bands"]


@dataclass
class NoDataPolicy:
    nodata_value: float = 65536
    rule: NoDataRule = "control-band"
    control_band_1based: int = 1
    compute_valid_before_nodata_remap: bool = True


@dataclass
class QARasterConfig:
    require_crs: bool = True
    require_geotransform: bool = True
    allow_rotated_geotransform: bool = False
    min_valid_ratio_global: float = 0.70


@dataclass
class QVectorConfig:
    fix_invalid_geometries: str = "none"  # none | make_valid_in_memory | buffer0_in_memory
    drop_empty: bool = True
    explode_multipolygons: bool = True
    keep_holes: bool = True
    holes_policy: str = "background"
    min_area_m2: float = 1000.0
    clip_to_raster_bounds: bool = True
    clip_geometry: bool = False


@dataclass
class QAConfig:
    raster: QARasterConfig = field(default_factory=QARasterConfig)
    vector: QVectorConfig = field(default_factory=QVectorConfig)


@dataclass
class AOIClipConfig:
    enabled: bool = True
    mode: str = "bbox"          # bbox | mask
    buffer_m: float = 0.0
    mask_outside: bool = False
    out_dir: Optional[str] = None
    compress: str = "DEFLATE"
    tiled: bool = True
    bigtiff: str = "if_needed"


@dataclass
class RasterPreprocessConfig:
    convert_dtype: str = "float32"
    nodata_to_value: float = 0.0
    compute_valid_ratio_before_nodata_zeroing: bool = True
    compute_band_stats_enabled: bool = True
    band_stats_method: str = "sample"
    sample_max_pixels: int = 5_000_000
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)
    save_path: Optional[str] = None


@dataclass
class NearNoDataConfig:
    enabled: bool = True
    ratio_target: float = 0.20
    nodata_frac_min: float = 0.05


@dataclass
class PatchingFilters:
    min_valid_ratio: float = 0.50
    min_mask_ratio: float = 0.03
    max_mask_ratio: float = 0.90
    neg_max_mask_ratio: float = 0.01


@dataclass
class PatchingConfig:
    patch_size_px: int = 512
    train_crop_px: int = 256
    pad_px: int = 16
    target_patches_per_dataset: int = 800

    sampling_mode: str = "mixed"
    weight_center: float = 0.6
    weight_boundary: float = 0.4
    samples_per_feature: int = 1

    negatives_enabled: bool = True
    negatives_ratio: float = 0.15
    negatives_min_distance_to_fields_m: float = 15.0

    near_nodata: NearNoDataConfig = field(default_factory=NearNoDataConfig)
    filters: PatchingFilters = field(default_factory=PatchingFilters)


@dataclass
class BWBLConfig:
    background_value: int = 0
    skeleton_value: int = 1
    buffer_value: int = 2
    buffer_px: int = 3


@dataclass
class IgnoreZoneConfig:
    enabled: bool = True
    ignore_value: int = 255
    ignore_radius_px: int = 7
    apply_to_extent: bool = True


@dataclass
class NoDataIgnorePolicy:
    enabled: bool = True
    extent_ig_value: int = 255
    bwbl_ignore_value: int = 2


@dataclass
class LabelsConfig:
    build_extent: bool = True
    build_boundary_raw: bool = True
    build_boundary_bwbl: bool = True
    build_extent_ig: bool = True

    boundary_include_holes: bool = True

    bwbl: BWBLConfig = field(default_factory=BWBLConfig)
    ignore_zone: IgnoreZoneConfig = field(default_factory=IgnoreZoneConfig)
    nodata_ignore_policy: NoDataIgnorePolicy = field(default_factory=NoDataIgnorePolicy)


@dataclass
class SplitRatios:
    train: float = 0.80
    validation: float = 0.10
    test: float = 0.10


@dataclass
class SplitConfig:
    ratios: SplitRatios = field(default_factory=SplitRatios)
    unit: str = "by_field"
    seed: int = 123
    spatial_blocking_enabled: bool = False


@dataclass
class ExportFolders:
    img: str = "img"
    extent: str = "extent"
    extent_ig: str = "extent_ig"
    boundary_raw: str = "boundary_raw"
    boundary_bwbl: str = "boundary_bwbl"
    valid: str = "valid"
    meta: str = "meta"


@dataclass
class ExportStructure:
    train_dir: str = "../prep_data/train"
    validation_dir: str = "../prep_data/validation"
    test_dir: str = "../prep_data/test"


@dataclass
class ExportConfig:
    structure: ExportStructure = field(default_factory=ExportStructure)
    folders: ExportFolders = field(default_factory=ExportFolders)


@dataclass
class PerformanceConfig:
    num_workers: int = 6
    gdal_cache_mb: int = 1024
    progress: bool = True


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: Optional[str] = None


# ----------------------------
# Main config object
# ----------------------------
@dataclass
class Config:
    config_path: Path
    project_root: Path

    paths: Dict[str, Path]
    datasets: List[DatasetSpec]

    nodata_policy: NoDataPolicy
    qa: QAConfig
    aoi_clip: AOIClipConfig
    raster_preprocess: RasterPreprocessConfig
    patching: PatchingConfig
    labels: LabelsConfig
    split: SplitConfig
    export: ExportConfig
    performance: PerformanceConfig
    logging: LoggingConfig

    raw: Dict[str, Any]


def _get(d: Dict[str, Any], keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_config(config_path: str | Path) -> Config:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.resolve()

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # paths (absolute)
    paths_raw = raw.get("paths", {})
    paths: Dict[str, Path] = {}
    for k, v in paths_raw.items():
        paths[k] = _abs_from(project_root, str(v))

    # datasets
    ds_list = raw.get("datasets", [])
    datasets: List[DatasetSpec] = []
    for ds in ds_list:
        root = _abs_from(project_root, ds["root"])
        raster = ds.get("raster", {})
        vector = ds.get("vector", {})
        datasets.append(
            DatasetSpec(
                name=ds["name"],
                root=root,
                raster_glob=list(raster.get("glob", ["*.tif", "*.tiff"])),
                raster_require_single=bool(raster.get("require_single_match", True)),
                vector_glob=list(vector.get("glob", ["*.shp", "*.gpkg"])),
                vector_require_single=bool(vector.get("require_single_match", True)),
                vector_layer=vector.get("layer", None),
                vector_id_field=vector.get("id_field", None),
            )
        )

    # nodata_policy
    nd_raw = raw.get("nodata_policy", raw.get("nodata", {}))
    nodata_policy = NoDataPolicy(
        nodata_value=float(nd_raw.get("nodata_value", nd_raw.get("value", 65536))),
        rule=str(nd_raw.get("rule", "control-band")),
        control_band_1based=int(nd_raw.get("control_band_1based", 1)),
        compute_valid_before_nodata_remap=bool(nd_raw.get("compute_valid_before_nodata_remap", True)),
    )

    # qa
    qa_raw = raw.get("qa", {}) or {}
    qa = QAConfig(
        raster=QARasterConfig(
            require_crs=bool(_get(qa_raw, ["raster", "require_crs"], True)),
            require_geotransform=bool(_get(qa_raw, ["raster", "require_geotransform"], True)),
            allow_rotated_geotransform=bool(_get(qa_raw, ["raster", "allow_rotated_geotransform"], False)),
            min_valid_ratio_global=float(_get(qa_raw, ["raster", "min_valid_ratio_global"], 0.70)),
        ),
        vector=QVectorConfig(
            fix_invalid_geometries=str(_get(qa_raw, ["vector", "fix_invalid_geometries"], "none")),
            drop_empty=bool(_get(qa_raw, ["vector", "drop_empty"], True)),
            explode_multipolygons=bool(_get(qa_raw, ["vector", "explode_multipolygons"], True)),
            keep_holes=bool(_get(qa_raw, ["vector", "keep_holes"], True)),
            holes_policy=str(_get(qa_raw, ["vector", "holes_policy"], "background")),
            min_area_m2=float(_get(qa_raw, ["vector", "min_area_m2"], 1000)),
            clip_to_raster_bounds=bool(_get(qa_raw, ["vector", "clip_to_raster_bounds"], True)),
            clip_geometry=bool(_get(qa_raw, ["vector", "clip_geometry"], False)),
        ),
    )

    # aoi_clip
    aoi_raw = raw.get("aoi_clip", {}) or {}
    aoi_clip = AOIClipConfig(
        enabled=bool(aoi_raw.get("enabled", True)),
        mode=str(aoi_raw.get("mode", "bbox")),
        buffer_m=float(aoi_raw.get("buffer_m", 0.0)),
        mask_outside=bool(aoi_raw.get("mask_outside", False)),
        out_dir=str(aoi_raw.get("out_dir")) if aoi_raw.get("out_dir") is not None else None,
        compress=str(aoi_raw.get("compress", "DEFLATE")),
        tiled=bool(aoi_raw.get("tiled", True)),
        bigtiff=str(aoi_raw.get("bigtiff", "if_needed")),
    )

    # raster_preprocess
    rp_raw = raw.get("raster_preprocess", {}) or {}
    stats_raw = rp_raw.get("compute_band_stats", {}) or {}
    raster_preprocess = RasterPreprocessConfig(
        convert_dtype=str(rp_raw.get("convert_dtype", "float32")),
        nodata_to_value=float(rp_raw.get("nodata_to_value", 0.0)),
        compute_valid_ratio_before_nodata_zeroing=bool(rp_raw.get("compute_valid_ratio_before_nodata_zeroing", True)),
        compute_band_stats_enabled=bool(stats_raw.get("enabled", True)),
        band_stats_method=str(stats_raw.get("method", "sample")),
        sample_max_pixels=int(stats_raw.get("sample_max_pixels", 5_000_000)),
        clip_percentiles=tuple(stats_raw.get("clip_percentiles", [1, 99])),
        save_path=str(stats_raw.get("save_path")) if stats_raw.get("save_path") is not None else None,
    )

    # patching
    p_raw = raw.get("patching", {}) or {}
    s_raw = p_raw.get("sampling", {}) or {}
    w_raw = s_raw.get("weights", {}) or {}
    f_raw = p_raw.get("filters", {}) or {}
    near_raw = s_raw.get("near_nodata", {}) or {}

    patching = PatchingConfig(
        patch_size_px=int(p_raw.get("patch_size_px", 512)),
        train_crop_px=int(p_raw.get("train_crop_px", 256)),
        pad_px=int(p_raw.get("pad_px", 16)),
        target_patches_per_dataset=int(p_raw.get("target_patches_per_dataset", 800)),
        sampling_mode=str(s_raw.get("mode", "mixed")),
        weight_center=float(w_raw.get("center", 0.6)),
        weight_boundary=float(w_raw.get("boundary", 0.4)),
        samples_per_feature=int(s_raw.get("samples_per_feature", 1)),
        negatives_enabled=bool(_get(s_raw, ["negatives", "enabled"], True)),
        negatives_ratio=float(_get(s_raw, ["negatives", "ratio"], 0.15)),
        negatives_min_distance_to_fields_m=float(_get(s_raw, ["negatives", "min_distance_to_fields_m"], 15)),
        near_nodata=NearNoDataConfig(
            enabled=bool(near_raw.get("enabled", True)),
            ratio_target=float(near_raw.get("ratio_target", 0.20)),
            nodata_frac_min=float(near_raw.get("nodata_frac_min", 0.05)),
        ),
        filters=PatchingFilters(
            min_valid_ratio=float(f_raw.get("min_valid_ratio", 0.50)),
            min_mask_ratio=float(f_raw.get("min_mask_ratio", 0.03)),
            max_mask_ratio=float(f_raw.get("max_mask_ratio", 0.90)),
            neg_max_mask_ratio=float(f_raw.get("neg_max_mask_ratio", 0.01)),
        ),
    )

    # labels
    l_raw = raw.get("labels", {}) or {}
    bw_raw = l_raw.get("bwbl", {}) or {}
    ig_raw = l_raw.get("ignore_zone", {}) or {}
    ndig_raw = l_raw.get("nodata_ignore_policy", {}) or {}

    labels = LabelsConfig(
        build_extent=bool(l_raw.get("build_extent", True)),
        build_boundary_raw=bool(l_raw.get("build_boundary_raw", True)),
        build_boundary_bwbl=bool(l_raw.get("build_boundary_bwbl", True)),
        build_extent_ig=bool(l_raw.get("build_extent_ig", True)),
        boundary_include_holes=bool(_get(l_raw, ["boundary", "include_holes"], True)),
        bwbl=BWBLConfig(
            background_value=int(bw_raw.get("background_value", 0)),
            skeleton_value=int(bw_raw.get("skeleton_value", 1)),
            buffer_value=int(bw_raw.get("buffer_value", 2)),
            buffer_px=int(bw_raw.get("buffer_px", 3)),
        ),
        ignore_zone=IgnoreZoneConfig(
            enabled=bool(ig_raw.get("enabled", True)),
            ignore_value=int(ig_raw.get("ignore_value", 255)),
            ignore_radius_px=int(ig_raw.get("ignore_radius_px", 7)),
            apply_to_extent=bool(ig_raw.get("apply_to_extent", True)),
        ),
        nodata_ignore_policy=NoDataIgnorePolicy(
            enabled=bool(ndig_raw.get("enabled", True)),
            extent_ig_value=int(ndig_raw.get("extent_ig_value", 255)),
            bwbl_ignore_value=int(ndig_raw.get("bwbl_ignore_value", 2)),
        ),
    )

    # split
    sp_raw = raw.get("split", {}) or {}
    split = SplitConfig(
        ratios=SplitRatios(
            train=float(_get(sp_raw, ["ratios", "train"], 0.80)),
            validation=float(_get(sp_raw, ["ratios", "validation"], 0.10)),
            test=float(_get(sp_raw, ["ratios", "test"], 0.10)),
        ),
        unit=str(sp_raw.get("unit", "by_field")),
        seed=int(sp_raw.get("seed", 123)),
        spatial_blocking_enabled=bool(_get(sp_raw, ["spatial_blocking", "enabled"], False)),
    )

    # export
    ex_raw = raw.get("export", {}) or {}
    st_raw = ex_raw.get("structure", {}) or {}
    fd_raw = ex_raw.get("folders", {}) or {}
    export = ExportConfig(
        structure=ExportStructure(
            train_dir=str(st_raw.get("train_dir", "../prep_data/train")),
            validation_dir=str(st_raw.get("validation_dir", "../prep_data/validation")),
            test_dir=str(st_raw.get("test_dir", "../prep_data/test")),
        ),
        folders=ExportFolders(
            img=str(fd_raw.get("img", "img")),
            extent=str(fd_raw.get("extent", "extent")),
            extent_ig=str(fd_raw.get("extent_ig", "extent_ig")),
            boundary_raw=str(fd_raw.get("boundary_raw", "boundary_raw")),
            boundary_bwbl=str(fd_raw.get("boundary_bwbl", "boundary_bwbl")),
            valid=str(fd_raw.get("valid", "valid")),
            meta=str(fd_raw.get("meta", "meta")),
        ),
    )

    # performance/logging
    perf_raw = raw.get("performance", {}) or {}
    performance = PerformanceConfig(
        num_workers=int(perf_raw.get("num_workers", 6)),
        gdal_cache_mb=int(perf_raw.get("gdal_cache_mb", 1024)),
        progress=bool(perf_raw.get("progress", True)),
    )

    log_raw = raw.get("logging", {}) or {}
    logging_cfg = LoggingConfig(
        level=str(log_raw.get("level", "INFO")),
        log_file=str(log_raw.get("log_file")) if log_raw.get("log_file") is not None else None,
    )

    return Config(
        config_path=config_path,
        project_root=project_root,
        paths=paths,
        datasets=datasets,
        nodata_policy=nodata_policy,
        qa=qa,
        aoi_clip=aoi_clip,
        raster_preprocess=raster_preprocess,
        patching=patching,
        labels=labels,
        split=split,
        export=export,
        performance=performance,
        logging=logging_cfg,
        raw=raw,
    )