from __future__ import annotations

import numpy as np

from net_train.infer.predict_aoi import _invalid_edge_band
from net_train.infer.tiling import blend_weights


def test_blend_weights_mean_is_uniform() -> None:
    ww = blend_weights(7, 11, mode="mean")
    assert ww.shape == (7, 11)
    assert np.allclose(ww, 1.0)


def test_blend_weights_gaussian_center_stronger_than_corner() -> None:
    ww = blend_weights(9, 9, mode="gaussian", gaussian_sigma=0.30, gaussian_min_weight=0.05)
    assert ww.shape == (9, 9)
    assert float(ww[4, 4]) > float(ww[0, 0])
    assert float(ww.min()) >= 0.05 - 1e-6


def test_invalid_edge_band_marks_valid_side_near_invalid() -> None:
    valid = np.ones((9, 9), dtype=np.uint8)
    valid[2:7, 2:7] = 0  # central invalid hole

    band = _invalid_edge_band(valid, radius_px=1)
    assert band.dtype == np.bool_
    assert np.any(band)

    # center stays invalid and not included in valid-side band
    assert not bool(band[4, 4])
    # immediate ring around the hole should be marked
    assert bool(band[1, 2]) or bool(band[2, 1]) or bool(band[1, 1])
