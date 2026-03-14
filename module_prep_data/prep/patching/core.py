from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point
from shapely.ops import unary_union

from .labels import (
    apply_nodata_ignore_policy,
    extent_and_boundaries_for_window,
    safe_window_centered,
)
from .manifest import build_dataset_summary, build_patch_meta
from .nodata import valid_mask_from_chip, valid_ratio_from_valid_mask
from .sampling import (
    build_field_ids,
    build_negative_distance_checker,
    jitter_point,
    sample_point_in_poly,
    sample_point_on_boundary,
)
from .writers import (
    ensure_dir,
    pixel_size_m,
    write_geotiff_mask,
    write_geotiff_multiband,
    write_json,
)


@dataclass
class PatchConfig:
    patch_size_px: int
    train_crop_px: int
    pad_px: int

    center_weight: float
    boundary_weight: float

    negatives_ratio: float
    negatives_min_dist_m: float

    min_valid_ratio: float
    min_mask_ratio: float
    max_mask_ratio: float
    neg_max_mask_ratio: float

    bwbl_buffer_px: int
    bwbl_background_value: int
    bwbl_skeleton_value: int
    bwbl_buffer_value: int

    build_extent: bool
    build_extent_ig: bool
    build_boundary_raw: bool
    build_boundary_bwbl: bool
    boundary_include_holes: bool

    ignore_radius_px: int
    ignore_enabled: bool
    ignore_value: int
    ignore_apply_to_extent: bool

    nodata_value: float = 65536
    nodata_rule: str = "control-band"
    control_band_1based: int = 1

    write_valid_mask: bool = True

    apply_nodata_ignore_policy: bool = True
    nodata_ignore_extent_value: int = 255
    nodata_ignore_bwbl_value: int = 2

    seed: int = 123
    target_patches: int = 800


def make_patches_for_dataset(
    raster_path: Path,
    vector_path: Path,
    out_root: Path,
    ds_name: str,
    cfg: PatchConfig,
    cleaned_vector_gpkg: Optional[Path] = None,
    vector_layer: Optional[str] = None,
    vector_id_field: Optional[str] = None,
) -> Path:
    """
    Writes patches into:
      out_root/<ds_name>/{img,extent,extent_ig,boundary_raw,boundary_bwbl,valid,meta}
    Returns path to manifest.json
    """
    rng = np.random.default_rng(cfg.seed)

    out_ds = out_root / ds_name
    img_dir = out_ds / "img"
    extent_dir = out_ds / "extent"
    extent_ig_dir = out_ds / "extent_ig"
    braw_dir = out_ds / "boundary_raw"
    bwbl_dir = out_ds / "boundary_bwbl"
    valid_dir = out_ds / "valid"
    meta_dir = out_ds / "meta"
    for d in [img_dir, extent_dir, extent_ig_dir, braw_dir, bwbl_dir, valid_dir, meta_dir]:
        ensure_dir(d)

    gdf = gpd.read_file(str(vector_path), layer=vector_layer) if vector_layer else gpd.read_file(str(vector_path))
    if gdf.empty:
        raise RuntimeError("Vector is empty")
    if gdf.crs is None:
        raise RuntimeError("Vector CRS missing")

    with rasterio.open(str(raster_path)) as ds:
        if ds.crs is None:
            raise RuntimeError("Raster CRS missing")
        raster_crs = ds.crs

        gdf = gdf.to_crs(raster_crs)
        gdf = gdf.explode(index_parts=False, ignore_index=True)
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()
        gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
        if gdf.empty:
            raise RuntimeError("No polygon features left after CRS/geometry filtering")

        try:
            sindex = gdf.sindex
        except Exception:
            sindex = None

        union_geom = unary_union([g for g in gdf.geometry.values if g is not None and not g.is_empty])
        pix_m = pixel_size_m(ds)

        target_total = int(cfg.target_patches)
        neg_ratio = float(cfg.negatives_ratio)
        pos_target = int(round(target_total * (1.0 - neg_ratio)))
        neg_target = int(target_total - pos_target)

        w_sum = max(1e-9, cfg.center_weight + cfg.boundary_weight)
        center_target = int(round(pos_target * (cfg.center_weight / w_sum)))
        boundary_target = int(pos_target - center_target)

        patch_w = patch_h = int(cfg.patch_size_px)
        if ds.width < patch_w or ds.height < patch_h:
            raise RuntimeError(f"Raster too small for patch_size={patch_w}: raster size is {ds.width}x{ds.height}")

        nfeat = len(gdf)
        if nfeat == 0:
            raise RuntimeError("No features after explode")

        field_ids, field_id_source = build_field_ids(ds_name=ds_name, gdf=gdf, vector_id_field=vector_id_field)

        written = 0
        written_center = 0
        written_boundary = 0
        written_neg = 0

        manifest_rows: List[dict] = []
        rejects = {"oob": 0, "valid": 0, "mask": 0, "neg_dist": 0, "neg_mask": 0, "other": 0}

        def try_write_patch(
            patch_id: str,
            win: Window,
            inside_mode: str,
            feat_i: Optional[int],
        ) -> bool:
            nonlocal written, written_center, written_boundary, written_neg

            chip = ds.read(window=win)
            valid_u8 = valid_mask_from_chip(
                chip,
                nodata_value=float(cfg.nodata_value),
                nodata_rule=str(cfg.nodata_rule),
                control_band_1based=int(cfg.control_band_1based),
            )
            vr = valid_ratio_from_valid_mask(valid_u8)
            if vr < cfg.min_valid_ratio:
                rejects["valid"] += 1
                return False

            extent, extent_ig, braw, bwbl, st = extent_and_boundaries_for_window(
                ds=ds,
                gdf=gdf,
                sindex=sindex,
                win=win,
                include_holes=cfg.boundary_include_holes,
                ignore_enabled=cfg.ignore_enabled,
                ignore_value=cfg.ignore_value,
                ignore_apply_to_extent=cfg.ignore_apply_to_extent,
                ignore_radius_px=cfg.ignore_radius_px,
                pad_px=cfg.pad_px,
                bwbl_buffer_px=cfg.bwbl_buffer_px,
                bwbl_background_value=cfg.bwbl_background_value,
                bwbl_skeleton_value=cfg.bwbl_skeleton_value,
                bwbl_buffer_value=cfg.bwbl_buffer_value,
            )
            mr = st["mask_ratio"]

            if inside_mode == "negative":
                if mr > cfg.neg_max_mask_ratio:
                    rejects["neg_mask"] += 1
                    return False
            else:
                if mr < cfg.min_mask_ratio or mr > cfg.max_mask_ratio:
                    rejects["mask"] += 1
                    return False

            if cfg.apply_nodata_ignore_policy:
                extent_ig, bwbl = apply_nodata_ignore_policy(
                    extent_ig=extent_ig,
                    bwbl=bwbl,
                    valid_u8=valid_u8,
                    extent_ignore_value=int(cfg.nodata_ignore_extent_value),
                    bwbl_ignore_value=int(cfg.nodata_ignore_bwbl_value),
                )

            base = patch_id
            img_path = img_dir / f"img_{base}.tif"
            extent_path = extent_dir / f"extent_{base}.tif"
            extent_ig_path = extent_ig_dir / f"extent_ig_{base}.tif"
            braw_path = braw_dir / f"boundary_raw_{base}.tif"
            bwbl_path = bwbl_dir / f"bwbl_{base}.tif"
            valid_path = valid_dir / f"valid_{base}.tif"
            meta_path = meta_dir / f"meta_{base}.json"

            extent_out = extent.astype(np.uint8) if cfg.build_extent else np.zeros_like(extent, dtype=np.uint8)
            extent_ig_out = extent_ig.astype(np.uint8) if cfg.build_extent_ig else np.zeros_like(extent_ig, dtype=np.uint8)
            braw_out = braw.astype(np.uint8) if cfg.build_boundary_raw else np.zeros_like(braw, dtype=np.uint8)
            bwbl_out = (
                bwbl.astype(np.uint8)
                if cfg.build_boundary_bwbl
                else np.full_like(bwbl, np.uint8(cfg.bwbl_background_value), dtype=np.uint8)
            )

            write_geotiff_multiband(img_path, chip, ds, win)
            write_geotiff_mask(extent_path, extent_out, ds, win)
            write_geotiff_mask(extent_ig_path, extent_ig_out, ds, win)
            write_geotiff_mask(braw_path, braw_out, ds, win)
            write_geotiff_mask(bwbl_path, bwbl_out, ds, win)
            if cfg.write_valid_mask:
                write_geotiff_mask(valid_path, valid_u8.astype(np.uint8), ds, win)

            nodata_frac = float(1.0 - vr)
            field_id = str(field_ids[int(feat_i)]) if feat_i is not None else None
            meta = build_patch_meta(
                ds_name=ds_name,
                patch_id=base,
                inside_mode=inside_mode,
                feat_i=feat_i,
                field_id=field_id,
                win=win,
                valid_ratio=float(vr),
                nodata_frac=float(nodata_frac),
                mask_ratio=float(mr),
                edge_ratio=float(st["edge_ratio"]),
                skeleton_ratio=float(st["skeleton_ratio"]),
                ignore_ratio=float(st["ignore_ratio"]),
                pixel_size_m=float(pix_m),
                train_crop_px=int(cfg.train_crop_px),
                pad_px=int(cfg.pad_px),
                nodata_value=float(cfg.nodata_value),
                nodata_rule=str(cfg.nodata_rule),
                control_band_1based=int(cfg.control_band_1based),
                labels_written={
                    "extent": bool(cfg.build_extent),
                    "extent_ig": bool(cfg.build_extent_ig),
                    "boundary_raw": bool(cfg.build_boundary_raw),
                    "boundary_bwbl": bool(cfg.build_boundary_bwbl),
                    "valid": bool(cfg.write_valid_mask),
                },
            )
            write_json(meta_path, meta)
            manifest_rows.append(meta)

            written += 1
            if inside_mode == "center":
                written_center += 1
            elif inside_mode == "boundary":
                written_boundary += 1
            else:
                written_neg += 1
            return True

        patch_idx = 0
        center_attempts = 0
        max_center_attempts = max(center_target * 50, 2000)
        while written_center < center_target and center_attempts < max_center_attempts:
            center_attempts += 1
            fi = int(rng.integers(0, nfeat))
            poly = gdf.geometry.iloc[int(fi)]
            p = sample_point_in_poly(poly, rng=rng)
            if p is None:
                rejects["other"] += 1
                continue
            row, col = ds.index(p.x, p.y)
            off = safe_window_centered(col, row, patch_w, patch_h, ds.width, ds.height)
            if off is None:
                rejects["oob"] += 1
                continue
            xoff, yoff = off
            win = Window(xoff, yoff, patch_w, patch_h)
            pid = f"{ds_name}_{patch_idx:06d}"
            patch_idx += 1
            _ = try_write_patch(pid, win, "center", int(fi))

        jitter_m = max(1.0, 0.10 * patch_w * pix_m)
        boundary_attempts = 0
        max_boundary_attempts = max(boundary_target * 50, 2000)
        while written_boundary < boundary_target and boundary_attempts < max_boundary_attempts:
            boundary_attempts += 1
            fi = int(rng.integers(0, nfeat))
            poly = gdf.geometry.iloc[int(fi)]
            p0 = sample_point_on_boundary(poly, rng=rng)
            if p0 is None:
                rejects["other"] += 1
                continue
            p = jitter_point(p0, rng=rng, jitter_m=jitter_m)
            row, col = ds.index(p.x, p.y)
            off = safe_window_centered(col, row, patch_w, patch_h, ds.width, ds.height)
            if off is None:
                rejects["oob"] += 1
                continue
            xoff, yoff = off
            win = Window(xoff, yoff, patch_w, patch_h)
            pid = f"{ds_name}_{patch_idx:06d}"
            patch_idx += 1
            _ = try_write_patch(pid, win, "boundary", int(fi))

        min_dist = float(cfg.negatives_min_dist_m)
        is_too_close = build_negative_distance_checker(ds, union_geom, min_dist_m=min_dist)

        neg_attempts = 0
        max_neg_attempts = max(neg_target * 50, 5000)
        col_min = int(patch_w // 2)
        col_max = int(ds.width - (patch_w - patch_w // 2))
        row_min = int(patch_h // 2)
        row_max = int(ds.height - (patch_h - patch_h // 2))

        while written_neg < neg_target and neg_attempts < max_neg_attempts:
            neg_attempts += 1
            col = int(rng.integers(col_min, col_max + 1))
            row = int(rng.integers(row_min, row_max + 1))

            x, y = ds.xy(row, col)
            p = Point(float(x), float(y))

            try:
                if is_too_close(p):
                    rejects["neg_dist"] += 1
                    continue
            except Exception:
                pass

            off = safe_window_centered(col, row, patch_w, patch_h, ds.width, ds.height)
            if off is None:
                rejects["oob"] += 1
                continue
            xoff, yoff = off
            win = Window(xoff, yoff, patch_w, patch_h)

            pid = f"{ds_name}_{patch_idx:06d}"
            patch_idx += 1
            ok = try_write_patch(pid, win, "negative", None)
            if not ok:
                continue

        summary = build_dataset_summary(
            ds_name=ds_name,
            raster_path=str(raster_path),
            vector_path=str(vector_path),
            vector_layer=vector_layer,
            vector_id_field=vector_id_field,
            field_id_source=field_id_source,
            written_total=int(written),
            written_center=int(written_center),
            written_boundary=int(written_boundary),
            written_negative=int(written_neg),
            target_total=int(target_total),
            pos_target=int(pos_target),
            center_target=int(center_target),
            boundary_target=int(boundary_target),
            neg_target=int(neg_target),
            center_attempts=int(center_attempts),
            boundary_attempts=int(boundary_attempts),
            neg_attempts=int(neg_attempts),
            rejects=rejects,
            nodata_value=float(cfg.nodata_value),
            nodata_rule=str(cfg.nodata_rule),
            control_band_1based=int(cfg.control_band_1based),
        )

        manifest_path = out_ds / "manifest.json"
        write_json(manifest_path, {"summary": summary, "patches": manifest_rows})

        if cleaned_vector_gpkg is not None:
            try:
                gdf.to_file(cleaned_vector_gpkg, driver="GPKG", layer="fields_raster_crs")
            except Exception:
                pass

        return manifest_path
