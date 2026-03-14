from __future__ import annotations

import numpy as np

from net_train.data.transforms import AugmentConfig, TrainAugmentor, near_invalid_band, random_crop


def test_near_invalid_band_marks_valid_side_border() -> None:
    valid = np.ones((7, 7), dtype=np.uint8)
    valid[2:5, 2:5] = 0

    band = near_invalid_band(valid, radius_px=1)
    assert band.dtype == np.bool_
    assert np.any(band)
    assert not bool(band[3, 3])  # invalid center stays excluded
    assert bool(band[1, 2]) or bool(band[2, 1]) or bool(band[1, 1])


def test_random_crop_near_invalid_bias_prefers_edge_region() -> None:
    rng = np.random.default_rng(42)
    img = np.zeros((1, 8, 8), dtype=np.float32)
    for y in range(8):
        for x in range(8):
            img[0, y, x] = float(10 * y + x)
    extent = np.zeros((8, 8), dtype=np.uint8)
    boundary = np.zeros((8, 8), dtype=np.uint8)
    valid = np.ones((8, 8), dtype=np.uint8)
    valid[:, :2] = 0  # clear invalid strip on the left

    crop, _, _ = random_crop(
        img=img,
        extent=extent,
        boundary=boundary,
        crop_size=4,
        rng=rng,
        attempts=32,
        valid_mask=valid,
        near_invalid_radius_px=1,
        min_near_invalid_pixels=2,
        near_invalid_bias_prob=1.0,  # enforce
    )
    assert crop.shape == (1, 4, 4)
    top_left = int(round(float(crop[0, 0, 0])))
    x0 = top_left % 10
    # left-most columns are invalid; near-invalid enforced crop should be close to that side.
    assert x0 <= 2


def test_train_augmentor_invalid_edge_sim_applies_valid_cutout() -> None:
    cfg = AugmentConfig(
        enabled=True,
        hflip=False,
        vflip=False,
        rotate90=False,
        invalid_edge_sim_enabled=True,
        invalid_edge_sim_prob=1.0,
        invalid_edge_sim_min_width_px=2,
        invalid_edge_sim_max_width_px=3,
        invalid_edge_sim_block_prob=1.0,
        invalid_edge_sim_max_area_ratio=0.25,
        invalid_edge_sim_zero_image=True,
    )
    aug = TrainAugmentor(cfg, seed=7, valid_channel_index=2)

    img = np.ones((3, 12, 12), dtype=np.float32)
    extent = np.zeros((12, 12), dtype=np.uint8)
    boundary = np.zeros((12, 12), dtype=np.uint8)

    img2, extent2, boundary2 = aug(img, extent, boundary)
    assert img2.shape == img.shape
    assert extent2.shape == extent.shape
    assert boundary2.shape == boundary.shape
    assert aug.last_invalid_edge_applied is True
    assert np.any(img2[2] == 0.0)  # valid channel has new invalid area
