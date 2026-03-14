from __future__ import annotations

import numpy as np

from net_train.data.transforms import random_crop


def test_random_crop_can_enforce_boundary_positive_pixels() -> None:
    rng = np.random.default_rng(123)

    img = np.zeros((2, 8, 8), dtype=np.float32)
    extent = np.zeros((8, 8), dtype=np.uint8)
    boundary = np.zeros((8, 8), dtype=np.uint8)

    extent[0:3, 0:3] = 1
    boundary[1:3, 1:3] = 1

    _, e_crop, b_crop = random_crop(
        img=img,
        extent=extent,
        boundary=boundary,
        crop_size=4,
        rng=rng,
        min_extent_pixels=1,
        min_boundary_pixels=1,
        attempts=32,
    )

    assert int((e_crop == 1).sum()) >= 1
    assert int((b_crop == 1).sum()) >= 1


def test_random_crop_fallback_prob_controls_best_effort_vs_random() -> None:
    img = np.arange(36, dtype=np.float32).reshape(1, 6, 6)
    extent = np.zeros((6, 6), dtype=np.uint8)
    boundary = np.zeros((6, 6), dtype=np.uint8)

    # Threshold is impossible, so function uses fallback path.
    rng_best = np.random.default_rng(123)
    crop_best, _, _ = random_crop(
        img=img,
        extent=extent,
        boundary=boundary,
        crop_size=3,
        rng=rng_best,
        min_extent_pixels=0,
        min_boundary_pixels=5,
        attempts=1,
        fallback_to_best_prob=1.0,
    )

    rng_random = np.random.default_rng(123)
    crop_random, _, _ = random_crop(
        img=img,
        extent=extent,
        boundary=boundary,
        crop_size=3,
        rng=rng_random,
        min_extent_pixels=0,
        min_boundary_pixels=5,
        attempts=1,
        fallback_to_best_prob=0.0,
    )

    assert not np.array_equal(crop_best, crop_random)
