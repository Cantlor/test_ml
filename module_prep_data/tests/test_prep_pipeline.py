from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import rasterio


def _read1(p: Path) -> np.ndarray:
    with rasterio.open(p) as ds:
        return ds.read(1)


def test_patches_all_has_required_layers():
    proj_root = Path(__file__).resolve().parents[2]  # my_project/
    patches_all = proj_root / "output_data" / "module_prep_data_work" / "patches_all"
    assert patches_all.exists(), f"patches_all missing: {patches_all}"

    ds_dirs = [p for p in patches_all.iterdir() if p.is_dir()]
    assert ds_dirs, "no dataset dirs in patches_all"

    for ds in ds_dirs:
        for sub in ["img", "extent_ig", "boundary_bwbl", "valid", "meta"]:
            assert (ds / sub).exists(), f"missing {sub} in {ds}"


def test_valid_mask_and_nodata_ignore_policy_on_sample():
    proj_root = Path(__file__).resolve().parents[2]
    patches_all = proj_root / "output_data" / "module_prep_data_work" / "patches_all"
    ds_dirs = [p for p in patches_all.iterdir() if p.is_dir()]
    assert ds_dirs, "no dataset dirs in patches_all"

    # sample a few from first dataset
    ds = ds_dirs[0]
    metas = sorted(glob.glob(str(ds / "meta" / "meta_*.json")))[:10]
    assert metas, f"no meta files in {ds}"

    ids = [Path(m).stem.split("meta_")[1] for m in metas]

    for pid in ids:
        p_valid = ds / "valid" / f"valid_{pid}.tif"
        p_extig = ds / "extent_ig" / f"extent_ig_{pid}.tif"
        p_bwbl = ds / "boundary_bwbl" / f"bwbl_{pid}.tif"

        assert p_valid.exists()
        assert p_extig.exists()
        assert p_bwbl.exists()

        valid = _read1(p_valid)
        extig = _read1(p_extig)
        bwbl = _read1(p_bwbl)

        assert set(np.unique(valid)).issubset({0, 1})
        assert set(np.unique(extig)).issubset({0, 1, 255})
        assert set(np.unique(bwbl)).issubset({0, 1, 2})

        invalid = (valid == 0)
        if invalid.any():
            assert np.all(extig[invalid] == 255)
            assert np.all(bwbl[invalid] == 2)


def test_prep_data_split_contains_valid():
    proj_root = Path(__file__).resolve().parents[2]
    prep_data = proj_root / "prep_data"
    assert prep_data.exists(), f"prep_data missing: {prep_data}"

    for split in ["train", "validation", "test"]:
        base = prep_data / split
        assert base.exists(), f"missing split dir: {base}"
        assert (base / "valid").exists(), f"missing valid/ in {base}"