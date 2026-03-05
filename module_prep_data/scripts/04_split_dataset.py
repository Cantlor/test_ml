from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from rich.console import Console

# run from module_prep_data/
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PatchRec:
    dataset: str
    patch_id: str
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
            feat_index = p.get("feat_index", None)
            inside_mode = p.get("inside_mode", "unknown")

            src_img = ds_dir / "img" / f"img_{patch_id}.tif"
            # берём extent_ig как финальный extent (0/1/255)
            src_extent_ig = ds_dir / "extent_ig" / f"extent_ig_{patch_id}.tif"
            src_braw = ds_dir / "boundary_raw" / f"boundary_raw_{patch_id}.tif"
            src_bwbl = ds_dir / "boundary_bwbl" / f"bwbl_{patch_id}.tif"
            # ✅ NEW: valid mask
            src_valid = ds_dir / "valid" / f"valid_{patch_id}.tif"
            src_meta = ds_dir / "meta" / f"meta_{patch_id}.json"

            for fp in [src_img, src_extent_ig, src_braw, src_bwbl, src_valid, src_meta]:
                if not fp.exists():
                    raise RuntimeError(f"missing file: {fp}")

            out.append(
                PatchRec(
                    dataset=dataset,
                    patch_id=patch_id,
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
    # by_field: всё с одинаковым feat_index держим вместе
    # negatives (feat_index=None) распределяем независимо
    if rec.feat_index is None:
        return f"neg::{rec.patch_id}"
    return f"field::{rec.dataset}::{rec.feat_index}"


def split_groups(
    recs: List[PatchRec],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[PatchRec]]:
    import numpy as np

    rng = np.random.default_rng(seed)

    groups: Dict[str, List[PatchRec]] = {}
    for r in recs:
        groups.setdefault(group_key(r), []).append(r)

    keys = list(groups.keys())
    rng.shuffle(keys)

    N = len(recs)
    rsum = float(train_ratio + val_ratio + test_ratio)
    if rsum <= 0:
        raise RuntimeError("Invalid split ratios: sum(train,val,test) must be > 0")

    # normalize
    tr = float(train_ratio / rsum)
    vr = float(val_ratio / rsum)

    tN = int(round(N * tr))
    vN = int(round(N * vr))

    splits = {"train": [], "validation": [], "test": []}
    counts = {"train": 0, "validation": 0, "test": 0}

    for k in keys:
        g = groups[k]
        gsz = len(g)

        if counts["train"] < tN:
            splits["train"].extend(g)
            counts["train"] += gsz
        elif counts["validation"] < vN:
            splits["validation"].extend(g)
            counts["validation"] += gsz
        else:
            splits["test"].extend(g)
            counts["test"] += gsz

    return splits


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    vals = [train_ratio, val_ratio, test_ratio]
    if any(v < 0 or v > 1 for v in vals):
        raise RuntimeError("Each ratio must be in [0, 1]")
    if sum(vals) <= 0:
        raise RuntimeError("At least one ratio must be > 0")


def link_or_copy(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            return
        dst.unlink()

    try:
        os.link(src, dst)  # hardlink
    except Exception:
        shutil.copy2(src, dst)


def dir_nonempty(p: Path) -> bool:
    return p.exists() and any(p.iterdir())


def wipe_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


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

    # Проверка, что не оставляем мешанину
    for split in ["train", "validation", "test"]:
        for sub in ["img", "extent", "boundary_raw", "boundary_bwbl", "valid", "meta"]:
            p = out_root / split / sub
            if dir_nonempty(p) and not args.overwrite:
                raise RuntimeError(
                    f"Destination not empty: {p}. Run with --overwrite to recreate prep_data/{split}."
                )

    if args.overwrite:
        for split in ["train", "validation", "test"]:
            wipe_dir(out_root / split)

    console.print("[bold]04_split_dataset (by_field)[/bold]")
    console.print(f"patches_all: {patches_all}")
    console.print(f"out_prep_data: {out_root}")
    console.print(f"ratios: train={args.train} val={args.val} test={args.test}  seed={args.seed}")
    console.print("NOTE: extent/ будет взят из extent_ig (0/1/255).")

    recs = collect_patches(patches_all)
    console.print(f"found patches: {len(recs)}")

    splits = split_groups(recs, args.train, args.val, args.test, args.seed)

    # Copy/link
    for split_name, items in splits.items():
        console.print(f"\n[bold]{split_name}[/bold] items={len(items)}")
        base = out_root / split_name

        for r in items:
            link_or_copy(r.src_img, base / "img" / f"img_{r.patch_id}.tif", args.overwrite)
            link_or_copy(r.src_extent_ig, base / "extent" / f"extent_{r.patch_id}.tif", args.overwrite)
            link_or_copy(r.src_braw, base / "boundary_raw" / f"boundary_raw_{r.patch_id}.tif", args.overwrite)
            link_or_copy(r.src_bwbl, base / "boundary_bwbl" / f"bwbl_{r.patch_id}.tif", args.overwrite)
            # ✅ NEW: valid
            link_or_copy(r.src_valid, base / "valid" / f"valid_{r.patch_id}.tif", args.overwrite)
            link_or_copy(r.src_meta, base / "meta" / f"meta_{r.patch_id}.json", args.overwrite)

    split_manifest = {
        "patches_all": str(patches_all),
        "out_prep_data": str(out_root),
        "seed": args.seed,
        "ratios": {"train": args.train, "validation": args.val, "test": args.test},
        "counts": {k: len(v) for k, v in splits.items()},
        "notes": {
            "extent_source": "extent_ig (0/1/255)",
            "valid_mask_included": True,
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, ensure_ascii=False, indent=2)

    console.print(f"\n[green]DONE[/green] wrote: {out_root / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())