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


@dataclass
class DatasetOptions:
    crop_size: int
    nodata_value: float | None
    extent_ignore_value: int
    boundary_ignore_value: int
    is_train: bool


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

    def _read_triplet(self, rec: SampleRecord) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with rasterio.open(rec.img_path) as ds:
            img = ds.read().astype(np.float32)

        with rasterio.open(rec.extent_path) as ds:
            extent = ds.read(1).astype(np.uint8)

        with rasterio.open(rec.boundary_bwbl_path) as ds:
            boundary = ds.read(1).astype(np.uint8)

        return img, extent, boundary

    def __getitem__(self, idx: int) -> Dict[str, object]:
        rec = self.records[idx]
        img, extent, boundary = self._read_triplet(rec)

        if self.options.is_train and self.options.crop_size > 0:
            img, extent, boundary = random_crop(
                img,
                extent,
                boundary,
                crop_size=int(self.options.crop_size),
                rng=self.rng,
            )

        if self.options.is_train:
            img, extent, boundary = self.augmentor(img, extent, boundary)

        img = normalize_image(
            img,
            self.norm_stats,
            nodata_value=self.options.nodata_value,
        )

        # Convert targets into training-ready tensors.
        # extent: values {0,1,255(ignore)}
        # boundary_bwbl: values {0,1,2(ignore)}
        x = torch.from_numpy(np.ascontiguousarray(img)).float()
        y_extent = torch.from_numpy(np.ascontiguousarray(extent)).long()
        y_boundary = torch.from_numpy(np.ascontiguousarray(boundary)).long()

        sample: Dict[str, object] = {
            "image": x,
            "extent": y_extent,
            "boundary": y_boundary,
            "patch_id": rec.patch_id,
            "dataset": rec.dataset,
        }
        return sample
