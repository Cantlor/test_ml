from __future__ import annotations

# Backward-compatible facade.
from .patching.core import PatchConfig, make_patches_for_dataset

__all__ = ["PatchConfig", "make_patches_for_dataset"]
