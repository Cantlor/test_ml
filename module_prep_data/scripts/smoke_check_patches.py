from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio


def _pick_patch_ids(ds_dir: Path, k: int) -> List[str]:
    metas = sorted(glob.glob(str(ds_dir / "meta" / "meta_*.json")))
    if not metas:
        # fallback: pick by valid files
        vs = sorted(glob.glob(str(ds_dir / "valid" / "valid_*.tif")))
        ids = [Path(p).stem.split("valid_")[1] for p in vs[:k]]
        return ids
    ids = [Path(p).stem.split("meta_")[1] for p in metas[:k]]
    return ids


def _read1(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read(1)


def _assert_unique_subset(name: str, arr: np.ndarray, allowed: Tuple[int, ...]) -> None:
    u = np.unique(arr)
    bad = [int(x) for x in u if int(x) not in allowed]
    if bad:
        raise AssertionError(f"{name}: unexpected values {bad}, allowed={allowed}, uniq={u}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_all", required=True, help=".../output_data/module_prep_data_work/patches_all")
    ap.add_argument("--dataset", default=None, help="if set, only check this dataset folder name")
    ap.add_argument("--k", type=int, default=10, help="how many patches to sample per dataset")
    args = ap.parse_args()

    root = Path(args.patches_all).resolve()
    if not root.exists():
        raise SystemExit(f"patches_all not found: {root}")

    ds_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if args.dataset:
        ds_dirs = [p for p in ds_dirs if p.name == args.dataset]
        if not ds_dirs:
            raise SystemExit(f"dataset not found: {args.dataset} in {root}")

    total_checked = 0
    total_with_invalid = 0

    for ds_dir in ds_dirs:
        print(f"\n[smoke] dataset: {ds_dir.name}")
        required_dirs = ["img", "extent_ig", "boundary_bwbl", "valid", "meta"]
        for d in required_dirs:
            if not (ds_dir / d).exists():
                raise AssertionError(f"missing dir: {ds_dir/d}")

        ids = _pick_patch_ids(ds_dir, args.k)
        if not ids:
            raise AssertionError(f"no patch ids found in {ds_dir}")

        for pid in ids:
            p_valid = ds_dir / "valid" / f"valid_{pid}.tif"
            p_extig = ds_dir / "extent_ig" / f"extent_ig_{pid}.tif"
            p_bwbl = ds_dir / "boundary_bwbl" / f"bwbl_{pid}.tif"

            if not p_valid.exists():
                raise AssertionError(f"missing: {p_valid}")
            if not p_extig.exists():
                raise AssertionError(f"missing: {p_extig}")
            if not p_bwbl.exists():
                raise AssertionError(f"missing: {p_bwbl}")

            valid = _read1(p_valid)
            extig = _read1(p_extig)
            bwbl = _read1(p_bwbl)

            _assert_unique_subset("valid", valid, (0, 1))
            _assert_unique_subset("extent_ig", extig, (0, 1, 255))
            _assert_unique_subset("bwbl", bwbl, (0, 1, 2))

            invalid = (valid == 0)
            if invalid.any():
                total_with_invalid += 1
                # ключевая проверка политики NoData-ignore
                if not np.all(extig[invalid] == 255):
                    u = np.unique(extig[invalid])
                    raise AssertionError(f"{ds_dir.name}:{pid} extent_ig on invalid must be 255, got uniq={u}")
                if not np.all(bwbl[invalid] == 2):
                    u = np.unique(bwbl[invalid])
                    raise AssertionError(f"{ds_dir.name}:{pid} bwbl on invalid must be 2, got uniq={u}")

            total_checked += 1

        print(f"[smoke] checked patches: {len(ids)} (invalid-present patches among sample: {total_with_invalid})")

    print(f"\n[smoke] DONE. total_checked={total_checked}, total_with_invalid={total_with_invalid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())