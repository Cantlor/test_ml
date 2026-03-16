from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from ..patching.budget import compute_patch_targets, estimate_auto_patch_budget_for_paths, stable_dataset_seed
from ..progress import iter_progress
from ..utils import ensure_dir, write_json


def _deferred_config_keys(cfg) -> List[str]:
    keys: List[str] = []
    if str(cfg.patching.patch_budget.mode).strip().lower() == "auto":
        keys.append("patching.patch_budget.*")
    if cfg.patching.near_nodata.enabled:
        keys.append("patching.sampling.near_nodata.*")
    if cfg.patching.samples_per_feature != 1:
        keys.append("patching.sampling.samples_per_feature")
    if str(cfg.patching.sampling_mode).strip().lower() != "mixed":
        keys.append("patching.sampling.mode")
    if cfg.raster_preprocess.compute_band_stats_enabled:
        keys.append("raster_preprocess.compute_band_stats.*")
    return keys


def _effective_budget_mode(cfg, n_override: Optional[int]) -> str:
    mode = str(cfg.patching.patch_budget.mode).strip().lower()
    if mode not in {"fixed", "auto"}:
        mode = "fixed"
    if n_override is not None:
        mode = "fixed"
    return mode


def _budget_report_path(work_dir: Path, budget_report_out: Optional[str | Path]) -> Path:
    if budget_report_out is not None:
        return Path(budget_report_out).resolve()
    return (work_dir / "patch_budget_report.json").resolve()


def run(
    config_path: str | Path,
    n_override: Optional[int] = None,
    seed_override: Optional[int] = None,
    dry_run_budget: bool = False,
    budget_report_out: Optional[str | Path] = None,
) -> int:
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
    budget_mode = _effective_budget_mode(cfg, n_override=n_override)

    work_dir = get_work_dir(cfg)
    out_root = (work_dir / "patches_all").resolve()

    check_manifest = load_check_inputs_manifest_required(cfg)
    aoi_manifest = load_aoi_manifest_required(cfg) if cfg.aoi_clip.enabled else None

    console.print("[bold]03_make_patches[/bold]")
    console.print(f"config: {cfg.config_path}")
    console.print(f"out_root: {out_root}")
    console.print(f"target_patches_per_dataset: {target}")
    console.print(
        "patch_budget: "
        f"mode={budget_mode} "
        f"min_valid_patches_to_keep={cfg.patching.patch_budget.min_valid_patches_to_keep} "
        f"max_patches_per_dataset={cfg.patching.patch_budget.max_patches_per_dataset}"
    )
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

    if dry_run_budget:
        report_path = _budget_report_path(work_dir=work_dir, budget_report_out=budget_report_out)
        console.print("[cyan]mode[/cyan]: dry-run budget preview (no patch files will be written)")

        show_progress = bool(cfg.performance.progress)
        datasets_report: List[Dict[str, Any]] = []

        checked = 0
        skipped_auto = 0
        skipped_effective = 0
        total_estimated_target = 0
        total_estimated_center = 0
        total_estimated_boundary = 0
        total_estimated_negative = 0
        total_effective_target = 0
        total_effective_center = 0
        total_effective_boundary = 0
        total_effective_negative = 0

        ds_iter = iter_progress(
            cfg.datasets,
            total=len(cfg.datasets),
            desc="preview-budget",
            unit="dataset",
            enabled=show_progress,
        )
        for ds in ds_iter:
            raster_path, raster_source, vector_path, vector_layer = resolve_patch_inputs_for_dataset(
                cfg=cfg,
                ds=ds,
                check_manifest=check_manifest,
                aoi_manifest=aoi_manifest,
            )

            budget_info = estimate_auto_patch_budget_for_paths(
                raster_path=raster_path,
                vector_path=vector_path,
                vector_layer=vector_layer,
                patch_size_px=int(patch_size_px),
                train_crop_px=int(train_crop_px),
                negatives_ratio=float(neg_ratio),
                center_weight=float(center_w),
                boundary_weight=float(boundary_w),
                capacity_overlap_factor=float(cfg.patching.patch_budget.capacity_overlap_factor),
                boundary_pixels_per_patch=float(cfg.patching.patch_budget.boundary_pixels_per_patch),
                min_valid_patches_to_keep=int(cfg.patching.patch_budget.min_valid_patches_to_keep),
                min_patches_per_dataset=int(cfg.patching.patch_budget.min_patches_per_dataset),
                max_patches_per_dataset=int(cfg.patching.patch_budget.max_patches_per_dataset),
                valid_ratio_sample_windows=int(cfg.patching.patch_budget.valid_ratio_sample_windows),
                nodata_value=float(cfg.nodata_policy.nodata_value),
                nodata_rule=str(cfg.nodata_policy.rule),
                control_band_1based=int(cfg.nodata_policy.control_band_1based),
                seed=stable_dataset_seed(seed, ds.name),
            )

            fixed_targets = compute_patch_targets(
                total_target=int(target),
                negatives_ratio=float(neg_ratio),
                center_weight=float(center_w),
                boundary_weight=float(boundary_w),
            )
            auto_targets = compute_patch_targets(
                total_target=int(budget_info.get("estimated_total_target", 0)),
                negatives_ratio=float(neg_ratio),
                center_weight=float(center_w),
                boundary_weight=float(boundary_w),
            )

            auto_skipped = bool(budget_info.get("skipped_due_to_low_capacity", False))
            auto_skip_reason = budget_info.get("skip_reason")

            if budget_mode == "auto":
                effective_targets = auto_targets
                effective_skipped = auto_skipped
                effective_skip_reason = auto_skip_reason
            else:
                effective_targets = fixed_targets
                effective_skipped = False
                effective_skip_reason = None

            valid_area_ratio = float(budget_info.get("valid_area_ratio", 0.0) or 0.0)
            capacity_limiters = budget_info.get("capacity_limiters", []) or []
            limiters_text = ",".join(str(x) for x in capacity_limiters) if capacity_limiters else "-"

            console.print(f"\n[bold]{ds.name}[/bold]")
            console.print(f"  patch_budget_mode={budget_mode} patch_size_px={patch_size_px}")
            console.print(
                "  "
                f"valid_area_ratio={valid_area_ratio:.4f} "
                f"estimated_total_capacity={int(budget_info.get('estimated_total_capacity', 0))} "
                f"estimated_total_target={int(budget_info.get('estimated_total_target', 0))}"
            )
            console.print(
                "  "
                f"estimated_targets(c/b/n)="
                f"{int(budget_info.get('estimated_center_target', 0))}/"
                f"{int(budget_info.get('estimated_boundary_target', 0))}/"
                f"{int(budget_info.get('estimated_negative_target', 0))}"
            )
            console.print(
                "  "
                f"skipped_due_to_low_capacity={auto_skipped} "
                f"skip_reason={auto_skip_reason or '-'} "
                f"capacity_limiters={limiters_text}"
            )
            if budget_mode == "fixed":
                console.print(
                    "  "
                    f"effective_fixed_target(c/b/n)="
                    f"{int(fixed_targets['center_target'])}/"
                    f"{int(fixed_targets['boundary_target'])}/"
                    f"{int(fixed_targets['neg_target'])} "
                    f"total={int(fixed_targets['target_total'])}"
                )

            checked += 1
            if auto_skipped:
                skipped_auto += 1
            if effective_skipped:
                skipped_effective += 1

            total_estimated_target += int(budget_info.get("estimated_total_target", 0) or 0)
            total_estimated_center += int(budget_info.get("estimated_center_target", 0) or 0)
            total_estimated_boundary += int(budget_info.get("estimated_boundary_target", 0) or 0)
            total_estimated_negative += int(budget_info.get("estimated_negative_target", 0) or 0)

            total_effective_target += int(effective_targets["target_total"])
            total_effective_center += int(effective_targets["center_target"])
            total_effective_boundary += int(effective_targets["boundary_target"])
            total_effective_negative += int(effective_targets["neg_target"])

            datasets_report.append(
                {
                    "dataset": ds.name,
                    "patch_budget_mode": budget_mode,
                    "patch_size_px": int(patch_size_px),
                    "raster_path": str(raster_path),
                    "raster_source": str(raster_source),
                    "vector_path": str(vector_path),
                    "vector_layer": vector_layer,
                    "auto_budget_info": budget_info,
                    "fixed_targets": fixed_targets,
                    "effective_targets": effective_targets,
                    "auto_skipped_due_to_low_capacity": bool(auto_skipped),
                    "auto_skip_reason": auto_skip_reason,
                    "effective_skipped_due_to_low_capacity": bool(effective_skipped),
                    "effective_skip_reason": effective_skip_reason,
                }
            )

        summary = {
            "datasets_checked": int(checked),
            "datasets_skipped_auto_estimate": int(skipped_auto),
            "datasets_skipped_effective": int(skipped_effective),
            "total_estimated_target": int(total_estimated_target),
            "total_estimated_center": int(total_estimated_center),
            "total_estimated_boundary": int(total_estimated_boundary),
            "total_estimated_negative": int(total_estimated_negative),
            "total_effective_target": int(total_effective_target),
            "total_effective_center": int(total_effective_center),
            "total_effective_boundary": int(total_effective_boundary),
            "total_effective_negative": int(total_effective_negative),
        }
        report_obj = {
            "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "config_path": str(cfg.config_path),
            "work_dir": str(work_dir.resolve()),
            "dry_run_budget": True,
            "patch_budget_mode": budget_mode,
            "patch_size_px": int(patch_size_px),
            "datasets": datasets_report,
            "summary": summary,
        }
        write_json(report_path, report_obj)

        console.print("\n[bold]budget preview summary[/bold]")
        console.print(
            f"datasets_checked={checked} "
            f"datasets_skipped_auto_estimate={skipped_auto} "
            f"datasets_skipped_effective={skipped_effective} "
            f"total_estimated_target={total_estimated_target} "
            f"total_effective_target={total_effective_target}"
        )
        console.print(
            "total_estimated(c/b/n)="
            f"{total_estimated_center}/{total_estimated_boundary}/{total_estimated_negative}"
        )
        console.print(
            "total_effective(c/b/n)="
            f"{total_effective_center}/{total_effective_boundary}/{total_effective_negative}"
        )
        console.print(f"[green]DONE[/green] budget report: {report_path}")
        return 0

    datasets_manifest: List[PatchesDatasetResult] = []
    ensure_dir(out_root)
    show_progress = bool(cfg.performance.progress)
    ds_iter = iter_progress(
        cfg.datasets,
        total=len(cfg.datasets),
        desc="make-patches",
        unit="dataset",
        enabled=show_progress,
    )
    for ds in ds_iter:
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
            show_progress=show_progress,
            seed=seed,
            target_patches=target,
            patch_budget_mode=budget_mode,
            min_valid_patches_to_keep=int(cfg.patching.patch_budget.min_valid_patches_to_keep),
            min_patches_per_dataset=int(cfg.patching.patch_budget.min_patches_per_dataset),
            max_patches_per_dataset=int(cfg.patching.patch_budget.max_patches_per_dataset),
            capacity_overlap_factor=float(cfg.patching.patch_budget.capacity_overlap_factor),
            boundary_pixels_per_patch=float(cfg.patching.patch_budget.boundary_pixels_per_patch),
            valid_ratio_sample_windows=int(cfg.patching.patch_budget.valid_ratio_sample_windows),
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
        summary: dict = {}
        try:
            with open(dataset_manifest_path, "r", encoding="utf-8") as f:
                summary = (json.load(f) or {}).get("summary", {}) or {}
        except Exception:
            summary = {}

        skipped = bool(summary.get("skipped_due_to_low_capacity", False))
        skip_reason = summary.get("skip_reason")
        status = "skipped" if skipped else "ok"
        message = str(skip_reason) if skipped and skip_reason else None

        if skipped:
            console.print(
                f"[yellow]SKIPPED[/yellow] {ds.name}: "
                f"{message or 'low estimated patch capacity'}"
            )
        else:
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
                status=status,
                message=message,
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
