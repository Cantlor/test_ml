from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from net_train.utils.io import read_json


@dataclass
class SampleRecord:
    split: str
    patch_id: str
    dataset: str
    meta_path: Path
    img_path: Path
    extent_path: Path
    boundary_bwbl_path: Path
    meta: Dict[str, Any]


@dataclass
class IndexResult:
    split: str
    records: List[SampleRecord]
    missing_files: List[str]


def _expected_paths(split_root: Path, patch_id: str) -> Dict[str, Path]:
    return {
        "img": split_root / "img" / f"img_{patch_id}.tif",
        "extent": split_root / "extent" / f"extent_{patch_id}.tif",
        "boundary_bwbl": split_root / "boundary_bwbl" / f"bwbl_{patch_id}.tif",
    }


def build_index_for_split(prep_data_root: Path, split_name: str) -> IndexResult:
    split_root = prep_data_root / split_name
    meta_dir = split_root / "meta"

    if not meta_dir.exists():
        return IndexResult(split=split_name, records=[], missing_files=[f"Missing directory: {meta_dir}"])

    records: List[SampleRecord] = []
    missing: List[str] = []

    for meta_path in sorted(meta_dir.glob("meta_*.json")):
        meta = read_json(meta_path)
        patch_id = str(meta.get("patch_id", "")).strip()
        dataset = str(meta.get("dataset", "unknown"))

        if not patch_id:
            missing.append(f"{meta_path}: missing patch_id")
            continue

        paths = _expected_paths(split_root, patch_id)
        miss_here = [f"{kind}: {p}" for kind, p in paths.items() if not p.exists()]
        if miss_here:
            missing.extend([f"{meta_path}: {m}" for m in miss_here])
            continue

        records.append(
            SampleRecord(
                split=split_name,
                patch_id=patch_id,
                dataset=dataset,
                meta_path=meta_path,
                img_path=paths["img"],
                extent_path=paths["extent"],
                boundary_bwbl_path=paths["boundary_bwbl"],
                meta=meta,
            )
        )

    return IndexResult(split=split_name, records=records, missing_files=missing)


def build_index(prep_data_root: Path, splits: Dict[str, str]) -> Dict[str, IndexResult]:
    out: Dict[str, IndexResult] = {}
    for alias, split_name in splits.items():
        out[alias] = build_index_for_split(prep_data_root=prep_data_root, split_name=split_name)
    return out
