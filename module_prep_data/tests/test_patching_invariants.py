from __future__ import annotations

import numpy as np

from prep.patching.labels import apply_nodata_ignore_policy
from prep.patching.nodata import valid_mask_from_chip


def test_valid_mask_control_band_rule():
    chip = np.array(
        [
            [[65536, 1], [1, 65536]],
            [[10, 10], [10, 10]],
        ],
        dtype=np.int32,
    )
    valid = valid_mask_from_chip(
        chip=chip,
        nodata_value=65536,
        nodata_rule="control-band",
        control_band_1based=1,
    )
    expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(valid, expected)


def test_valid_mask_all_bands_rule():
    chip = np.array(
        [
            [[65536, 1], [65536, 65536]],
            [[65536, 1], [2, 65536]],
        ],
        dtype=np.int32,
    )
    valid = valid_mask_from_chip(
        chip=chip,
        nodata_value=65536,
        nodata_rule="all-bands",
        control_band_1based=1,
    )
    expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(valid, expected)


def test_nodata_ignore_policy_semantics():
    extent_ig = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    bwbl = np.array([[0, 1], [2, 1]], dtype=np.uint8)
    valid = np.array([[1, 0], [1, 0]], dtype=np.uint8)

    e2, b2 = apply_nodata_ignore_policy(
        extent_ig=extent_ig,
        bwbl=bwbl,
        valid_u8=valid,
        extent_ignore_value=255,
        bwbl_ignore_value=2,
    )

    assert int(e2[0, 1]) == 255
    assert int(e2[1, 1]) == 255
    assert int(b2[0, 1]) == 2
    assert int(b2[1, 1]) == 2
