from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import rasterio

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    torch = None
    Dataset = object

from net_train.data.index import SampleRecord
from net_train.data.stats import NormalizationStats, normalize_image
from net_train.data.transforms import AugmentConfig, TrainAugmentor, near_invalid_band, random_crop


def compute_valid_mask(
    img_c_hw: np.ndarray,
    nodata_value: float | None,
    rule: str = "control-band",
    control_band_1based: int = 1,
) -> np.ndarray:
    """
    Returns uint8 mask [H,W] with values {0,1}.
    rule:
      - "control-band": NoData if img[control_band]==nodata_value
      - "all-bands":   NoData if all bands equal nodata_value
    """
    h, w = img_c_hw.shape[1], img_c_hw.shape[2]
    if nodata_value is None:
        return np.ones((h, w), dtype=np.uint8)

    if rule == "all-bands":
        is_nodata = np.all(img_c_hw == nodata_value, axis=0)
    else:
        b = int(control_band_1based) - 1
        b = max(0, min(b, img_c_hw.shape[0] - 1))
        is_nodata = (img_c_hw[b] == nodata_value)

    valid = (~is_nodata).astype(np.uint8)
    return valid


@dataclass
class DatasetOptions:
    crop_size: int
    num_bands: int
    nodata_value: float | None
    extent_ignore_value: int
    boundary_ignore_value: int
    is_train: bool

    # --- new / optional (backward-compatible defaults) ---
    add_valid_channel: bool = True
    nodata_rule: str = "control-band"          # "control-band" | "all-bands"
    control_band_1based: int = 1               # for "control-band"
    crop_attempts: int = 1
    crop_min_extent_pixels: int = 0
    crop_min_boundary_pixels: int = 0
    crop_fallback_to_best_prob: float = 1.0
    crop_near_invalid_enabled: bool = False
    crop_near_invalid_prob: float = 0.0
    crop_min_near_invalid_pixels: int = 0
    crop_near_invalid_radius_px: int = 2
    diag_near_invalid_radius_px: int = 2


class PatchDataset(Dataset):
    """Patch-level dataset for multitask extent + boundary training."""

    def __init__(
        self,
        records: List[SampleRecord],
        norm_stats: NormalizationStats,
        options: DatasetOptions,
        augment_cfg: Optional[AugmentConfig] = None,
        seed: int = 123,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to use PatchDataset")

        self.records = records
        self.norm_stats = norm_stats
        self.options = options
        self.rng = np.random.default_rng(seed)
        valid_channel_index = int(options.num_bands) if options.add_valid_channel else None
        self.augmentor = TrainAugmentor(
            augment_cfg or AugmentConfig(enabled=False),
            seed=seed,
            valid_channel_index=valid_channel_index,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _read_quad(self, rec: SampleRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # img: [C,H,W] float32
        with rasterio.open(rec.img_path) as ds:
            img = ds.read().astype(np.float32)
        expected_bands = int(self.options.num_bands)
        if img.shape[0] < expected_bands:
            raise RuntimeError(f"{rec.img_path}: expected at least {expected_bands} bands, got {img.shape[0]}")
        img = img[:expected_bands]

        # extent: [H,W] uint8 (0/1/255)
        with rasterio.open(rec.extent_path) as ds:
            extent = ds.read(1).astype(np.uint8)

        # boundary (BWBL): [H,W] uint8 (0/1/2)
        with rasterio.open(rec.boundary_bwbl_path) as ds:
            boundary = ds.read(1).astype(np.uint8)

        # valid: try from record, else compute from img via nodata policy
        valid_path = (
            getattr(rec, "valid_path", None)
            or getattr(rec, "valid_mask_path", None)
            or getattr(rec, "valid", None)
        )
        if valid_path is not None:
            with rasterio.open(valid_path) as ds:
                valid_raw = ds.read(1)
            uniq_valid = np.unique(valid_raw)
            if not np.all(np.isin(uniq_valid, [0, 1])):
                preview = [int(v) for v in uniq_valid[:8].tolist()]
                raise RuntimeError(
                    f"{valid_path}: valid mask must contain only {{0,1}}, got values like {preview}"
                )
            valid = valid_raw.astype(np.uint8)
        else:
            valid = compute_valid_mask(
                img,
                nodata_value=self.options.nodata_value,
                rule=self.options.nodata_rule,
                control_band_1based=self.options.control_band_1based,
            )

        if valid.shape != extent.shape:
            src = str(valid_path) if valid_path is not None else "<computed-from-image>"
            raise RuntimeError(f"{src}: valid shape {valid.shape} != extent shape {extent.shape}")

        img_hw = (img.shape[1], img.shape[2])
        if extent.shape != img_hw:
            raise RuntimeError(f"{rec.extent_path}: extent shape {extent.shape} != image shape {img_hw}")
        if boundary.shape != img_hw:
            raise RuntimeError(f"{rec.boundary_bwbl_path}: boundary shape {boundary.shape} != image shape {img_hw}")
        if valid.shape != img_hw:
            src = str(valid_path) if valid_path is not None else "<computed-from-image>"
            raise RuntimeError(f"{src}: valid shape {valid.shape} != image shape {img_hw}")

        extent_ignore = int(self.options.extent_ignore_value)
        boundary_ignore = int(self.options.boundary_ignore_value)
        bad_extent = np.unique(extent[~np.isin(extent, [0, 1, extent_ignore])])
        if bad_extent.size > 0:
            preview = [int(v) for v in bad_extent[:8].tolist()]
            raise RuntimeError(
                f"{rec.extent_path}: unexpected extent values {preview} "
                f"(allowed: 0, 1, {extent_ignore})"
            )
        bad_boundary = np.unique(boundary[~np.isin(boundary, [0, 1, boundary_ignore])])
        if bad_boundary.size > 0:
            preview = [int(v) for v in bad_boundary[:8].tolist()]
            raise RuntimeError(
                f"{rec.boundary_bwbl_path}: unexpected boundary values {preview} "
                f"(allowed: 0, 1, {boundary_ignore})"
            )

        return img, extent, boundary, valid

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        img8, extent, boundary, valid = self._read_quad(rec)
        synthetic_invalid_applied = 0.0

        # Build temporary image for geometric ops: [9,H,W] = img8 + valid
        # (we will normalize only first 8 channels later)
        if self.options.add_valid_channel:
            img = np.concatenate([img8, valid[None, :, :].astype(np.float32)], axis=0)
        else:
            img = img8

        # --- crop ---
        if self.options.is_train and self.options.crop_size > 0:
            img, extent, boundary = random_crop(
                img,
                extent,
                boundary,
                crop_size=int(self.options.crop_size),
                rng=self.rng,
                min_extent_pixels=int(self.options.crop_min_extent_pixels),
                min_boundary_pixels=int(self.options.crop_min_boundary_pixels),
                attempts=int(self.options.crop_attempts),
                fallback_to_best_prob=float(self.options.crop_fallback_to_best_prob),
                valid_mask=(valid if self.options.crop_near_invalid_enabled else None),
                near_invalid_radius_px=int(self.options.crop_near_invalid_radius_px),
                min_near_invalid_pixels=(
                    int(self.options.crop_min_near_invalid_pixels)
                    if self.options.crop_near_invalid_enabled
                    else 0
                ),
                near_invalid_bias_prob=(
                    float(self.options.crop_near_invalid_prob)
                    if self.options.crop_near_invalid_enabled
                    else 0.0
                ),
            )

        # --- augment ---
        if self.options.is_train:
            img, extent, boundary = self.augmentor(img, extent, boundary)
            synthetic_invalid_applied = 1.0 if bool(self.augmentor.last_invalid_edge_applied) else 0.0

        # Split back: img8 + valid (and re-binarize valid after possible aug noise)
        if self.options.add_valid_channel:
            nb = int(self.options.num_bands)
            img8 = img[:nb]
            valid_f = img[nb]
            valid = (valid_f >= 0.5).astype(np.uint8)
        else:
            img8 = img
            valid = compute_valid_mask(
                img8,
                nodata_value=self.options.nodata_value,
                rule=self.options.nodata_rule,
                control_band_1based=self.options.control_band_1based,
            )

        # Normalize ONLY the spectral bands.
        img8 = normalize_image(
            img8,
            self.norm_stats,
            nodata_value=self.options.nodata_value,
            valid_mask=valid,
        )

        # Enforce NoData ignore policy on targets (guard-rail):
        # valid=0 => extent ignore=255, boundary ignore=2
        if self.options.extent_ignore_value is not None:
            extent = extent.copy()
            extent[valid == 0] = np.uint8(self.options.extent_ignore_value)

        if self.options.boundary_ignore_value is not None:
            boundary = boundary.copy()
            boundary[valid == 0] = np.uint8(self.options.boundary_ignore_value)

        diag_radius = max(1, int(self.options.diag_near_invalid_radius_px))
        near_invalid = near_invalid_band(valid, radius_px=diag_radius)
        near_invalid_ratio = float(near_invalid.mean(dtype=np.float64))
        valid_ratio = float(valid.mean(dtype=np.float64))

        # Final model input: [9,H,W] (or [8,H,W] if disabled)
        if self.options.add_valid_channel:
            x_np = np.concatenate([img8, valid[None, :, :].astype(np.float32)], axis=0)
        else:
            x_np = img8

        expected_channels = int(self.options.num_bands + (1 if self.options.add_valid_channel else 0))
        if x_np.shape[0] != expected_channels:
            raise RuntimeError(
                f"{rec.patch_id}: model input channels={x_np.shape[0]} "
                f"!= expected={expected_channels} (num_bands={self.options.num_bands}, "
                f"add_valid_channel={self.options.add_valid_channel})"
            )

        # Tensors
        x = torch.from_numpy(np.ascontiguousarray(x_np)).float()
        y_extent = torch.from_numpy(np.ascontiguousarray(extent)).long()
        y_boundary = torch.from_numpy(np.ascontiguousarray(boundary)).long()
        v = torch.from_numpy(np.ascontiguousarray(valid)).float()

        sample: Dict[str, object] = {
            "image": x,
            "extent": y_extent,
            "boundary": y_boundary,
            "valid": v,  # удобно для дебага/инференса, можно не использовать в loss
            "patch_id": rec.patch_id,
            "dataset": rec.dataset,
            "near_invalid_ratio": float(near_invalid_ratio),
            "valid_ratio": float(valid_ratio),
            "synthetic_invalid_applied": float(synthetic_invalid_applied),
        }
        return sample
