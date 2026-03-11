from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from prep.config import load_config
from prep.utils import find_single_by_globs, ensure_dir
from prep.patches import PatchConfig, make_patches_for_dataset


def _resolve_aoi_raster(cfg, ds) -> Path:
    """
    Prefer AOI raster if aoi_clip.enabled and <out_dir>/<ds.name>_aoi.tif exists.
    Else fallback to raw raster matched by glob.
    """
    if cfg.aoi_clip.enabled and cfg.aoi_clip.out_dir:
        out_dir = Path(cfg.aoi_clip.out_dir)
        if not out_dir.is_absolute():
            out_dir = (cfg.project_root / out_dir).resolve()
        p = out_dir / f"{ds.name}_aoi.tif"
        if p.exists() and p.is_file():
            return p

    raster_path, matches = find_single_by_globs(ds.root, ds.raster_glob, ds.raster_require_single)
    if raster_path is None:
        raise RuntimeError(f"{ds.name}: expected single raster, got {len(matches)}")
    return raster_path


def _resolve_vector_for_patching(cfg, ds, work_dir: Path) -> Path:
    """
    Prefer prepared vector from 01_check_inputs.py if available:
      <work_dir>/<ds.name>_vector_prepared.gpkg
    Fallback to raw vector matched by config glob.
    """
    prepared = Path(work_dir) / f"{ds.name}_vector_prepared.gpkg"
    if prepared.exists() and prepared.is_file():
        return prepared

    vector_path, matches = find_single_by_globs(ds.root, ds.vector_glob, ds.vector_require_single)
    if vector_path is None:
        raise RuntimeError(f"{ds.name}: expected single vector, got {len(matches)}")
    return vector_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "prep_config.yaml"))
    ap.add_argument("--n", type=int, default=None, help="target patches per dataset (override config/default=800)")
    ap.add_argument("--seed", type=int, default=None, help="override seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    console = Console()

    # patching params (typed)
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

    # filters (ВАЖНО: min_valid_ratio должен быть не 0.90, иначе near-NoData патчи не пройдут)
    min_valid = float(cfg.patching.filters.min_valid_ratio)
    min_mask = float(cfg.patching.filters.min_mask_ratio)
    max_mask = float(cfg.patching.filters.max_mask_ratio)
    neg_max_mask = float(cfg.patching.filters.neg_max_mask_ratio)

    # labels (typed)
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

    # seed/target
    seed = int(cfg.split.seed)
    if args.seed is not None:
        seed = int(args.seed)

    target_default = int(cfg.patching.target_patches_per_dataset)
    target = int(args.n) if args.n is not None else target_default

    # staging output dir
    work_dir = cfg.paths.get("work_dir", (cfg.project_root / "../output_data/module_prep_data_work").resolve())
    out_root = Path(work_dir) / "patches_all"
    ensure_dir(out_root)

    console.print(f"[bold]03_make_patches[/bold]")
    console.print(f"config: {cfg.config_path}")
    console.print(f"out_root: {out_root}")
    console.print(f"target_patches_per_dataset: {target}")
    console.print(f"negatives_ratio: {neg_ratio} (enabled={neg_enabled})")
    console.print(
        f"labels: extent={build_extent} extent_ig={build_extent_ig} boundary_raw={build_boundary_raw} "
        f"boundary_bwbl={build_boundary_bwbl} include_holes={boundary_include_holes}"
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
        f"NoData policy (for later valid_mask stage): value={cfg.nodata_policy.nodata_value} "
        f"rule={cfg.nodata_policy.rule} control_band_1based={cfg.nodata_policy.control_band_1based}"
    )
    console.print(
        f"filters: min_valid={min_valid} min_mask={min_mask} max_mask={max_mask} neg_max_mask={neg_max_mask}"
    )

    for ds in cfg.datasets:
        raster_path = _resolve_aoi_raster(cfg, ds)
        vector_path = _resolve_vector_for_patching(cfg, ds, work_dir=Path(work_dir))

        console.print(f"\n[bold]{ds.name}[/bold]")
        console.print(f"  raster: {raster_path}")
        console.print(f"  vector: {vector_path}" + (f" (layer={ds.vector_layer})" if ds.vector_layer else ""))

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

            # ✅ ВАЖНО: прокидываем NoData policy из конфига
            nodata_value=float(cfg.nodata_policy.nodata_value),
            nodata_rule=str(cfg.nodata_policy.rule),
            control_band_1based=int(cfg.nodata_policy.control_band_1based),

            # ✅ ВАЖНО: пишем valid-mask и применяем ignore policy по NoData
            write_valid_mask=True,
            apply_nodata_ignore_policy=nodata_ignore_enabled,
            nodata_ignore_extent_value=nodata_ignore_extent_value,
            nodata_ignore_bwbl_value=nodata_ignore_bwbl_value,

            seed=seed,
            target_patches=target,
        )
        cleaned_vec_path = Path(work_dir) / f"{ds.name}_vector_raster_crs.gpkg"
        manifest = make_patches_for_dataset(
            raster_path=raster_path,
            vector_path=vector_path,
            out_root=out_root,
            ds_name=ds.name,
            cfg=pcfg,
            cleaned_vector_gpkg=cleaned_vec_path,
            vector_layer=ds.vector_layer,  # важно для GPKG
            vector_id_field=ds.vector_id_field,
        )

        console.print(f"[green]DONE[/green] manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
