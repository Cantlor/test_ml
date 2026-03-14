from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from rich.console import Console

from ..artifacts import (
    get_work_dir,
    load_aoi_manifest_required,
    load_check_inputs_manifest_required,
    patches_manifest_path,
    resolve_patch_inputs_for_dataset,
)
from ..config import load_config
from ..manifests import PatchesDatasetResult, PatchesManifest
from ..patches import PatchConfig, make_patches_for_dataset
from ..utils import ensure_dir


def _deferred_config_keys(cfg) -> List[str]:
    keys: List[str] = []
    if cfg.patching.near_nodata.enabled:
        keys.append("patching.sampling.near_nodata.*")
    if cfg.patching.samples_per_feature != 1:
        keys.append("patching.sampling.samples_per_feature")
    if str(cfg.patching.sampling_mode).strip().lower() != "mixed":
        keys.append("patching.sampling.mode")
    if cfg.raster_preprocess.compute_band_stats_enabled:
        keys.append("raster_preprocess.compute_band_stats.*")
    return keys


def run(config_path: str | Path, n_override: Optional[int] = None, seed_override: Optional[int] = None) -> int:
    cfg = load_config(config_path)
    console = Console()

    patch_size_px = int(cfg.patching.patch_size_px)
    train_crop_px = int(cfg.patching.train_crop_px)
    pad_px = int(cfg.patching.pad_px)

    if patch_size_px <= 0:
        raise RuntimeError("patching.patch_size_px must be > 0")
    if train_crop_px <= 0 or train_crop_px > patch_size_px:
        train_crop_px = patch_size_px
    if pad_px < 0:
        pad_px = 0
    if pad_px * 2 >= patch_size_px:
        raise RuntimeError("patching.pad_px is too large for patch_size_px")

    center_w = float(cfg.patching.weight_center)
    boundary_w = float(cfg.patching.weight_boundary)

    neg_enabled = bool(cfg.patching.negatives_enabled)
    neg_ratio = float(cfg.patching.negatives_ratio) if neg_enabled else 0.0
    neg_min_dist_m = float(cfg.patching.negatives_min_distance_to_fields_m)

    min_valid = float(cfg.patching.filters.min_valid_ratio)
    min_mask = float(cfg.patching.filters.min_mask_ratio)
    max_mask = float(cfg.patching.filters.max_mask_ratio)
    neg_max_mask = float(cfg.patching.filters.neg_max_mask_ratio)

    bwbl_background_value = int(cfg.labels.bwbl.background_value)
    bwbl_skeleton_value = int(cfg.labels.bwbl.skeleton_value)
    bwbl_buffer_value = int(cfg.labels.bwbl.buffer_value)
    bwbl_buffer_px = int(cfg.labels.bwbl.buffer_px)

    build_extent = bool(cfg.labels.build_extent)
    build_extent_ig = bool(cfg.labels.build_extent_ig)
    build_boundary_raw = bool(cfg.labels.build_boundary_raw)
    build_boundary_bwbl = bool(cfg.labels.build_boundary_bwbl)
    boundary_include_holes = bool(cfg.labels.boundary_include_holes)

    ignore_enabled = bool(cfg.labels.ignore_zone.enabled)
    ignore_value = int(cfg.labels.ignore_zone.ignore_value)
    ignore_apply_to_extent = bool(cfg.labels.ignore_zone.apply_to_extent)
    ignore_radius_px = int(cfg.labels.ignore_zone.ignore_radius_px) if ignore_enabled else 0

    nodata_ignore_enabled = bool(cfg.labels.nodata_ignore_policy.enabled)
    nodata_ignore_extent_value = int(cfg.labels.nodata_ignore_policy.extent_ig_value)
    nodata_ignore_bwbl_value = int(cfg.labels.nodata_ignore_policy.bwbl_ignore_value)

    seed = int(cfg.split.seed) if seed_override is None else int(seed_override)
    target = int(cfg.patching.target_patches_per_dataset) if n_override is None else int(n_override)

    work_dir = get_work_dir(cfg)
    out_root = work_dir / "patches_all"
    ensure_dir(out_root)

    check_manifest = load_check_inputs_manifest_required(cfg)
    aoi_manifest = load_aoi_manifest_required(cfg) if cfg.aoi_clip.enabled else None

    console.print("[bold]03_make_patches[/bold]")
    console.print(f"config: {cfg.config_path}")
    console.print(f"out_root: {out_root}")
    console.print(f"target_patches_per_dataset: {target}")
    console.print(f"negatives_ratio: {neg_ratio} (enabled={neg_enabled})")
    console.print(
        f"labels: extent={build_extent} extent_ig={build_extent_ig} "
        f"boundary_raw={build_boundary_raw} boundary_bwbl={build_boundary_bwbl} include_holes={boundary_include_holes}"
    )
    console.print(
        f"nodata_ignore_policy: enabled={nodata_ignore_enabled} "
        f"extent_ig_value={nodata_ignore_extent_value} bwbl_ignore_value={nodata_ignore_bwbl_value}"
    )
    if nodata_ignore_enabled and nodata_ignore_bwbl_value != bwbl_buffer_value:
        console.print(
            "[yellow]WARNING[/yellow] nodata_ignore_policy.bwbl_ignore_value "
            f"({nodata_ignore_bwbl_value}) != labels.bwbl.buffer_value ({bwbl_buffer_value})."
        )
    console.print(
        f"NoData policy: value={cfg.nodata_policy.nodata_value} "
        f"rule={cfg.nodata_policy.rule} control_band_1based={cfg.nodata_policy.control_band_1based}"
    )

    datasets_manifest: List[PatchesDatasetResult] = []
    for ds in cfg.datasets:
        raster_path, raster_source, vector_path, vector_layer = resolve_patch_inputs_for_dataset(
            cfg=cfg,
            ds=ds,
            check_manifest=check_manifest,
            aoi_manifest=aoi_manifest,
        )

        console.print(f"\n[bold]{ds.name}[/bold]")
        console.print(f"  raster: {raster_path} (source={raster_source})")
        console.print(f"  vector: {vector_path}" + (f" (layer={vector_layer})" if vector_layer else ""))

        pcfg = PatchConfig(
            patch_size_px=patch_size_px,
            train_crop_px=train_crop_px,
            pad_px=pad_px,
            center_weight=center_w,
            boundary_weight=boundary_w,
            negatives_ratio=neg_ratio,
            negatives_min_dist_m=neg_min_dist_m,
            min_valid_ratio=min_valid,
            min_mask_ratio=min_mask,
            max_mask_ratio=max_mask,
            neg_max_mask_ratio=neg_max_mask,
            bwbl_buffer_px=bwbl_buffer_px,
            bwbl_background_value=bwbl_background_value,
            bwbl_skeleton_value=bwbl_skeleton_value,
            bwbl_buffer_value=bwbl_buffer_value,
            build_extent=build_extent,
            build_extent_ig=build_extent_ig,
            build_boundary_raw=build_boundary_raw,
            build_boundary_bwbl=build_boundary_bwbl,
            boundary_include_holes=boundary_include_holes,
            ignore_radius_px=ignore_radius_px,
            ignore_enabled=ignore_enabled,
            ignore_value=ignore_value,
            ignore_apply_to_extent=ignore_apply_to_extent,
            nodata_value=float(cfg.nodata_policy.nodata_value),
            nodata_rule=str(cfg.nodata_policy.rule),
            control_band_1based=int(cfg.nodata_policy.control_band_1based),
            write_valid_mask=True,
            apply_nodata_ignore_policy=nodata_ignore_enabled,
            nodata_ignore_extent_value=nodata_ignore_extent_value,
            nodata_ignore_bwbl_value=nodata_ignore_bwbl_value,
            seed=seed,
            target_patches=target,
        )
        cleaned_vec_path = work_dir / f"{ds.name}_vector_raster_crs.gpkg"
        dataset_manifest_path = make_patches_for_dataset(
            raster_path=raster_path,
            vector_path=vector_path,
            out_root=out_root,
            ds_name=ds.name,
            cfg=pcfg,
            cleaned_vector_gpkg=cleaned_vec_path,
            vector_layer=vector_layer,
            vector_id_field=ds.vector_id_field,
        )
        console.print(f"[green]DONE[/green] dataset manifest: {dataset_manifest_path}")

        datasets_manifest.append(
            PatchesDatasetResult(
                dataset=ds.name,
                raster_path=str(raster_path),
                raster_source=str(raster_source),
                vector_path=str(vector_path),
                vector_layer=vector_layer,
                vector_id_field=ds.vector_id_field,
                dataset_manifest_path=str(dataset_manifest_path),
                cleaned_vector_raster_crs_path=str(cleaned_vec_path) if cleaned_vec_path.exists() else None,
                output_dataset_dir=str((out_root / ds.name).resolve()),
                status="ok",
                message=None,
            )
        )

    top_manifest = PatchesManifest.new(
        config_path=cfg.config_path,
        work_dir=work_dir,
        patches_all_root=out_root,
        datasets=datasets_manifest,
        deferred_config_keys=_deferred_config_keys(cfg),
    )
    mpath = top_manifest.save(patches_manifest_path(cfg))
    console.print(f"\n[green]DONE[/green] patches manifest: {mpath}")
    return 0
