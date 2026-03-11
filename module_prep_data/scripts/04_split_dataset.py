from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console

# run from module_prep_data/
ROOT = Path(__file__).resolve().parents[1]

SPLITS = ("train", "validation", "test")
SUBDIRS = ("img", "extent", "boundary_raw", "boundary_bwbl", "valid", "meta")
COPY_RULES = (
    ("src_img", "img", "img_{patch_id}.tif"),
    ("src_extent_ig", "extent", "extent_{patch_id}.tif"),
    ("src_braw", "boundary_raw", "boundary_raw_{patch_id}.tif"),
    ("src_bwbl", "boundary_bwbl", "bwbl_{patch_id}.tif"),
    ("src_valid", "valid", "valid_{patch_id}.tif"),
    ("src_meta", "meta", "meta_{patch_id}.json"),
)


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


def load_manifest(ds_dir: Path) -> List[dict]:
    mpath = ds_dir / "manifest.json"
    if not mpath.exists():
        raise RuntimeError(f"manifest not found: {mpath}")
    with open(mpath, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj.get("patches", [])


def collect_patches(patches_all_root: Path) -> List[PatchRec]:
    out: List[PatchRec] = []
    for ds_dir in sorted(patches_all_root.iterdir()):
        if not ds_dir.is_dir():
            continue

        patches = load_manifest(ds_dir)
        dataset = ds_dir.name

        for p in patches:
            patch_id = p["patch_id"]
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
                    dataset=dataset,
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
    # by_field: сначала стабильный field_id из manifest.
    # fallback: feat_index, если field_id отсутствует.
    # negatives (feat_index=None) держим как отдельные элементы.
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
    return (float(train_ratio / s), float(val_ratio / s), float(test_ratio / s))


def link_or_copy(src: Path, dst: Path, overwrite: bool) -> bool:
    """Returns True if file was linked/copied, False if skipped because dst exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return False
        dst.unlink()

    try:
        os.link(src, dst)  # hardlink
    except Exception:
        shutil.copy2(src, dst)
    return True


def wipe_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def ensure_tree(out_root: Path) -> None:
    for split in SPLITS:
        for sub in SUBDIRS:
            (out_root / split / sub).mkdir(parents=True, exist_ok=True)


def scan_existing_assignments(out_root: Path) -> Dict[str, str]:
    """
    patch_id -> split, based on prep_data/*/meta/meta_<patch_id>.json
    """
    out: Dict[str, str] = {}
    for split in SPLITS:
        meta_dir = out_root / split / "meta"
        if not meta_dir.exists():
            continue

        for mp in sorted(meta_dir.glob("meta_*.json")):
            stem = mp.stem
            if not stem.startswith("meta_"):
                continue
            patch_id = stem[len("meta_") :]
            prev = out.get(patch_id)
            if prev is not None and prev != split:
                raise RuntimeError(
                    f"Patch {patch_id} exists in multiple splits: {prev} and {split}."
                )
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
    """
    Assign only NEW patches; existing assignments are preserved.
    If a new group shares group_key with existing records, pin it to existing split.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    tr, vr, ter = _normalized_ratios(train_ratio, val_ratio, test_ratio)

    existing_counts = {s: 0 for s in SPLITS}
    for s in existing_patch_to_split.values():
        if s in existing_counts:
            existing_counts[s] += 1

    # group->split from already assigned records
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
                split = max(
                    SPLITS,
                    key=lambda s: (deficits[s], -counts[s], split_rank[s]),
                )
            else:
                split = min(
                    SPLITS,
                    key=lambda s: (counts[s], split_order[s]),
                )

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


def count_meta_per_split(out_root: Path) -> Dict[str, int]:
    return {
        split: len(list((out_root / split / "meta").glob("meta_*.json")))
        for split in SPLITS
    }


def copy_record(rec: PatchRec, split_root: Path, overwrite: bool) -> bool:
    copied = False
    for src_attr, subdir, name_tpl in COPY_RULES:
        src = getattr(rec, src_attr)
        dst = split_root / subdir / name_tpl.format(patch_id=rec.patch_id)
        copied |= link_or_copy(src, dst, overwrite=overwrite)
    return copied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_all", default=str((ROOT / "../output_data/module_prep_data_work/patches_all").resolve()))
    ap.add_argument("--out_prep_data", default=str((ROOT / "../prep_data").resolve()))
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    validate_ratios(args.train, args.val, args.test)

    console = Console()

    patches_all = Path(args.patches_all).resolve()
    out_root = Path(args.out_prep_data).resolve()

    if not patches_all.exists():
        raise RuntimeError(f"patches_all not found: {patches_all}")

    if args.overwrite:
        for split in SPLITS:
            wipe_dir(out_root / split)

    ensure_tree(out_root)

    console.print("[bold]04_split_dataset (by_field, append-safe)[/bold]")
    console.print(f"patches_all: {patches_all}")
    console.print(f"out_prep_data: {out_root}")
    console.print(f"mode: {'overwrite' if args.overwrite else 'append'}")
    console.print(f"ratios: train={args.train} val={args.val} test={args.test}  seed={args.seed}")
    console.print("NOTE: extent/ берётся из extent_ig (0/1/255).")

    recs = collect_patches(patches_all)
    console.print(f"found patches in patches_all: {len(recs)}")

    existing_assignments = {} if args.overwrite else scan_existing_assignments(out_root)
    console.print(f"already present in prep_data: {len(existing_assignments)}")

    assigned_new, assign_info = assign_new_records(
        recs=recs,
        existing_patch_to_split=existing_assignments,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )

    copied_counts = {s: 0 for s in SPLITS}

    for split_name in SPLITS:
        items = assigned_new[split_name]
        console.print(f"\n[bold]{split_name}[/bold] new_items={len(items)}")
        base = out_root / split_name

        for r in items:
            if copy_record(r, base, overwrite=args.overwrite):
                copied_counts[split_name] += 1

    final_counts = count_meta_per_split(out_root)

    split_manifest = {
        "patches_all": str(patches_all),
        "out_prep_data": str(out_root),
        "seed": args.seed,
        "mode": "overwrite" if args.overwrite else "append",
        "ratios": {"train": args.train, "validation": args.val, "test": args.test},
        "assign_info": assign_info,
        "copied_new_counts": copied_counts,
        "final_meta_counts": final_counts,
        "notes": {
            "extent_source": "extent_ig (0/1/255)",
            "valid_mask_included": True,
            "append_behavior": "existing prep_data assignments are preserved; only new patches are assigned and copied",
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]DONE[/green] wrote: {out_root / 'split_manifest.json'}")
    console.print(f"final counts (meta): {final_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
