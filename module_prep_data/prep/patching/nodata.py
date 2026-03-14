from __future__ import annotations

import numpy as np


def valid_mask_from_chip(
    chip: np.ndarray,
    nodata_value: float,
    nodata_rule: str,
    control_band_1based: int,
) -> np.ndarray:
    """
    chip: (C,H,W) any numeric dtype
    returns uint8 mask (H,W): 1 valid, 0 nodata
    """
    rule = (nodata_rule or "control-band").strip().lower()
    nd = nodata_value

    if rule == "control-band":
        b = int(control_band_1based) - 1
        if b < 0 or b >= chip.shape[0]:
            raise RuntimeError(
                f"control_band_1based out of range: {control_band_1based} for chip with C={chip.shape[0]}"
            )
        valid = chip[b] != nd
        return valid.astype(np.uint8)

    if rule == "all-bands":
        all_nd = np.all(chip == nd, axis=0)
        valid = ~all_nd
        return valid.astype(np.uint8)

    raise RuntimeError(f"Unknown nodata_rule: {nodata_rule}")


def valid_ratio_from_valid_mask(valid_u8: np.ndarray) -> float:
    return float(valid_u8.mean())
