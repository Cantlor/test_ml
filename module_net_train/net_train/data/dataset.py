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
from net_train.data.transforms import AugmentConfig, TrainAugmentor, random_crop


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
        self.augmentor = TrainAugmentor(augment_cfg or AugmentConfig(enabled=False), seed=seed)

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
                valid = ds.read(1).astype(np.uint8)
            # sanitize just in case
            valid = (valid > 0).astype(np.uint8)
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

        return img, extent, boundary, valid

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        img8, extent, boundary, valid = self._read_quad(rec)

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
            )

        # --- augment ---
        if self.options.is_train:
            img, extent, boundary = self.augmentor(img, extent, boundary)

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

        # Final model input: [9,H,W] (or [8,H,W] if disabled)
        if self.options.add_valid_channel:
            x_np = np.concatenate([img8, valid[None, :, :].astype(np.float32)], axis=0)
        else:
            x_np = img8

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
        }
        return sample
