from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rich.console import Console

from ..artifacts import (
    export_folders_from_cfg,
    export_split_roots_from_cfg,
    load_patches_manifest_required_from_path,
    patches_manifest_path as default_patches_manifest_path,
    split_manifest_path as default_split_manifest_path,
)
from ..config import load_config
from ..manifests import PatchesManifest, SplitManifest


SPLITS = ("train", "validation", "test")


@dataclass
class PatchRec:
    dataset: str
    patch_id: str
    field_id: str | None
    feat_index: int | None
    inside_mode: str
    src_img: Path
    src_extent_ig: Path
    src_braw: Path
    src_bwbl: Path
    src_valid: Path
    src_meta: Path


def load_dataset_manifest(path: Path) -> List[dict]:
    if not path.exists():
        raise RuntimeError(f"dataset patch manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    patches = obj.get("patches", [])
    if not isinstance(patches, list):
        raise RuntimeError(f"{path}: 'patches' must be list")
    return patches


def collect_patches(top_manifest: PatchesManifest) -> List[PatchRec]:
    out: List[PatchRec] = []
    for ds_item in top_manifest.datasets:
        ds_name = ds_item.dataset
        ds_dir = Path(ds_item.output_dataset_dir).resolve()
        rows = load_dataset_manifest(Path(ds_item.dataset_manifest_path).resolve())

        for p in rows:
            patch_id = str(p["patch_id"])
            field_id = p.get("field_id", None)
            feat_index = p.get("feat_index", None)
            inside_mode = p.get("inside_mode", "unknown")

            src_img = ds_dir / "img" / f"img_{patch_id}.tif"
            src_extent_ig = ds_dir / "extent_ig" / f"extent_ig_{patch_id}.tif"
            src_braw = ds_dir / "boundary_raw" / f"boundary_raw_{patch_id}.tif"
            src_bwbl = ds_dir / "boundary_bwbl" / f"bwbl_{patch_id}.tif"
            src_valid = ds_dir / "valid" / f"valid_{patch_id}.tif"
            src_meta = ds_dir / "meta" / f"meta_{patch_id}.json"

            for fp in [src_img, src_extent_ig, src_braw, src_bwbl, src_valid, src_meta]:
                if not fp.exists():
                    raise RuntimeError(f"missing file: {fp}")

            out.append(
                PatchRec(
                    dataset=ds_name,
                    patch_id=patch_id,
                    field_id=(str(field_id) if field_id is not None else None),
                    feat_index=int(feat_index) if feat_index is not None else None,
                    inside_mode=str(inside_mode),
                    src_img=src_img,
                    src_extent_ig=src_extent_ig,
                    src_braw=src_braw,
                    src_bwbl=src_bwbl,
                    src_valid=src_valid,
                    src_meta=src_meta,
                )
            )
    return out


def group_key(rec: PatchRec) -> str:
    if rec.field_id:
        return f"field::{rec.dataset}::{rec.field_id}"
    if rec.feat_index is None:
        return f"neg::{rec.patch_id}"
    return f"field::{rec.dataset}::{rec.feat_index}"


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    vals = [train_ratio, val_ratio, test_ratio]
    if any(v < 0 or v > 1 for v in vals):
        raise RuntimeError("Each ratio must be in [0, 1]")
    if sum(vals) <= 0:
        raise RuntimeError("At least one ratio must be > 0")


def _normalized_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    s = float(train_ratio + val_ratio + test_ratio)
    if s <= 0:
        raise RuntimeError("Invalid split ratios")
    return float(train_ratio / s), float(val_ratio / s), float(test_ratio / s)


def link_or_copy(src: Path, dst: Path, overwrite: bool) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return False
        dst.unlink()

    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)
    return True


def wipe_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def ensure_tree(split_roots: Dict[str, Path], folders: Dict[str, str]) -> None:
    base = ["img", "extent", "boundary_raw", "boundary_bwbl", "valid", "meta"]
    if folders.get("extent_ig") and folders["extent_ig"] != folders["extent"]:
        base.append("extent_ig")

    for split in SPLITS:
        root = split_roots[split]
        for key in base:
            (root / folders[key]).mkdir(parents=True, exist_ok=True)


def scan_existing_assignments(split_roots: Dict[str, Path], meta_folder: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for split in SPLITS:
        meta_dir = split_roots[split] / meta_folder
        if not meta_dir.exists():
            continue
        for mp in sorted(meta_dir.glob("meta_*.json")):
            stem = mp.stem
            if not stem.startswith("meta_"):
                continue
            patch_id = stem[len("meta_") :]
            prev = out.get(patch_id)
            if prev is not None and prev != split:
                raise RuntimeError(f"Patch {patch_id} exists in multiple splits: {prev} and {split}.")
            out[patch_id] = split
    return out


def assign_new_records(
    recs: List[PatchRec],
    existing_patch_to_split: Dict[str, str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dict[str, List[PatchRec]], Dict[str, object]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    tr, vr, ter = _normalized_ratios(train_ratio, val_ratio, test_ratio)

    existing_counts = {s: 0 for s in SPLITS}
    for s in existing_patch_to_split.values():
        if s in existing_counts:
            existing_counts[s] += 1

    group_existing_split: Dict[str, str] = {}
    for r in recs:
        sp = existing_patch_to_split.get(r.patch_id)
        if sp is None:
            continue
        gk = group_key(r)
        if gk not in group_existing_split:
            group_existing_split[gk] = sp

    new_recs = [r for r in recs if r.patch_id not in existing_patch_to_split]
    groups_new: Dict[str, List[PatchRec]] = {}
    for r in new_recs:
        groups_new.setdefault(group_key(r), []).append(r)

    keys = list(groups_new.keys())
    rng.shuffle(keys)

    total_after = int(len(existing_patch_to_split) + len(new_recs))
    tgt_train = int(round(total_after * tr))
    tgt_val = int(round(total_after * vr))
    tgt_test = int(total_after - tgt_train - tgt_val)
    targets = {"train": tgt_train, "validation": tgt_val, "test": tgt_test}

    counts = dict(existing_counts)
    out: Dict[str, List[PatchRec]] = {"train": [], "validation": [], "test": []}

    pinned_groups = 0
    split_rank = {"train": 2, "validation": 1, "test": 0}
    split_order = {"train": 0, "validation": 1, "test": 2}

    for gk in keys:
        g = groups_new[gk]
        gsz = len(g)

        pinned = group_existing_split.get(gk)
        if pinned in out:
            split = pinned
            pinned_groups += 1
        else:
            deficits = {s: int(targets[s] - counts[s]) for s in SPLITS}
            if any(v > 0 for v in deficits.values()):
                split = max(SPLITS, key=lambda s: (deficits[s], -counts[s], split_rank[s]))
            else:
                split = min(SPLITS, key=lambda s: (counts[s], split_order[s]))

        out[split].extend(g)
        counts[split] += gsz

    info = {
        "existing_total": int(len(existing_patch_to_split)),
        "new_total": int(len(new_recs)),
        "total_after": int(total_after),
        "targets": targets,
        "existing_counts": existing_counts,
        "final_expected_counts": counts,
        "groups_new": int(len(groups_new)),
        "groups_pinned_to_existing_split": int(pinned_groups),
    }
    return out, info


def count_meta_per_split(split_roots: Dict[str, Path], meta_folder: str) -> Dict[str, int]:
    return {split: len(list((split_roots[split] / meta_folder).glob("meta_*.json"))) for split in SPLITS}


def copy_record(rec: PatchRec, split_root: Path, folders: Dict[str, str], overwrite: bool) -> bool:
    copied = False
    mapping = [
        (rec.src_img, folders["img"], f"img_{rec.patch_id}.tif"),
        (rec.src_extent_ig, folders["extent"], f"extent_{rec.patch_id}.tif"),
        (rec.src_braw, folders["boundary_raw"], f"boundary_raw_{rec.patch_id}.tif"),
        (rec.src_bwbl, folders["boundary_bwbl"], f"bwbl_{rec.patch_id}.tif"),
        (rec.src_valid, folders["valid"], f"valid_{rec.patch_id}.tif"),
        (rec.src_meta, folders["meta"], f"meta_{rec.patch_id}.json"),
    ]
    if folders.get("extent_ig") and folders["extent_ig"] != folders["extent"]:
        mapping.append((rec.src_extent_ig, folders["extent_ig"], f"extent_ig_{rec.patch_id}.tif"))

    for src, subdir, name in mapping:
        dst = split_root / subdir / name
        copied |= link_or_copy(src, dst, overwrite=overwrite)
    return copied


def _resolve_patches_manifest_path(cfg, patches_manifest_override: Optional[str | Path], patches_all_override: Optional[str | Path]) -> Path:
    if patches_manifest_override is not None:
        return Path(patches_manifest_override).resolve()
    if patches_all_override is not None:
        p = Path(patches_all_override).resolve().parent / "patches_manifest.json"
        return p
    return default_patches_manifest_path(cfg).resolve()


def _resolve_split_roots(cfg, out_prep_data_override: Optional[str | Path]) -> Dict[str, Path]:
    if out_prep_data_override is None:
        return export_split_roots_from_cfg(cfg)
    root = Path(out_prep_data_override).resolve()
    return {split: root / split for split in SPLITS}


def _deferred_config_keys(cfg) -> List[str]:
    keys: List[str] = []
    if cfg.split.spatial_blocking_enabled:
        keys.append("split.spatial_blocking.enabled")
    return keys


def run(
    config_path: str | Path,
    patches_manifest_override: Optional[str | Path] = None,
    patches_all_override: Optional[str | Path] = None,
    out_prep_data_override: Optional[str | Path] = None,
    seed_override: Optional[int] = None,
    train_override: Optional[float] = None,
    val_override: Optional[float] = None,
    test_override: Optional[float] = None,
    overwrite: bool = False,
) -> int:
    cfg = load_config(config_path)
    if str(cfg.split.unit).strip().lower() != "by_field":
        raise RuntimeError(
            f"Only split.unit='by_field' is implemented in this stage. "
            f"Got split.unit='{cfg.split.unit}'."
        )

    seed = int(cfg.split.seed if seed_override is None else seed_override)
    train_ratio = float(cfg.split.ratios.train if train_override is None else train_override)
    val_ratio = float(cfg.split.ratios.validation if val_override is None else val_override)
    test_ratio = float(cfg.split.ratios.test if test_override is None else test_override)
    validate_ratios(train_ratio, val_ratio, test_ratio)

    pm_path = _resolve_patches_manifest_path(
        cfg=cfg,
        patches_manifest_override=patches_manifest_override,
        patches_all_override=patches_all_override,
    )
    top_manifest = load_patches_manifest_required_from_path(pm_path)

    split_roots = _resolve_split_roots(cfg, out_prep_data_override=out_prep_data_override)
    folders = export_folders_from_cfg(cfg)

    if overwrite:
        for split in SPLITS:
            wipe_dir(split_roots[split])
    ensure_tree(split_roots, folders)

    console = Console()
    console.print("[bold]04_split_dataset (by_field, append-safe, config-driven)[/bold]")
    console.print(f"config: {cfg.config_path}")
    console.print(f"patches_manifest: {pm_path}")
    console.print(f"mode: {'overwrite' if overwrite else 'append'}")
    console.print(f"ratios: train={train_ratio} val={val_ratio} test={test_ratio}  seed={seed}")
    console.print("NOTE: extent target is exported from extent_ig (ignore-aware extent contract).")

    recs = collect_patches(top_manifest)
    console.print(f"found patches in manifests: {len(recs)}")

    existing_assignments = {} if overwrite else scan_existing_assignments(split_roots, meta_folder=folders["meta"])
    console.print(f"already present in export roots: {len(existing_assignments)}")

    assigned_new, assign_info = assign_new_records(
        recs=recs,
        existing_patch_to_split=existing_assignments,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    copied_counts = {s: 0 for s in SPLITS}
    for split_name in SPLITS:
        items = assigned_new[split_name]
        console.print(f"\n[bold]{split_name}[/bold] new_items={len(items)}")
        base = split_roots[split_name]
        for r in items:
            if copy_record(r, base, folders=folders, overwrite=overwrite):
                copied_counts[split_name] += 1

    final_counts = count_meta_per_split(split_roots, meta_folder=folders["meta"])

    if out_prep_data_override is not None:
        split_manifest_out = (Path(out_prep_data_override).resolve() / "split_manifest.json").resolve()
    else:
        split_manifest_out = default_split_manifest_path(cfg).resolve()

    m = SplitManifest.new(
        config_path=cfg.config_path,
        patches_manifest_path=pm_path,
        mode="overwrite" if overwrite else "append",
        seed=seed,
        ratios={"train": train_ratio, "validation": val_ratio, "test": test_ratio},
        split_roots=split_roots,
        export_folders=folders,
        assign_info=assign_info,
        copied_new_counts=copied_counts,
        final_meta_counts=final_counts,
        notes={
            "extent_source": "extent_ig (0/1/255)",
            "valid_mask_included": True,
            "append_behavior": "existing assignments are preserved; only new patches are assigned and copied",
            "split_unit": "by_field",
        },
        deferred_config_keys=_deferred_config_keys(cfg),
    )
    m.save(split_manifest_out)

    console.print(f"\n[green]DONE[/green] wrote: {split_manifest_out}")
    console.print(f"final counts (meta): {final_counts}")
    return 0
