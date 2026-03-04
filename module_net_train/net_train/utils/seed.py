from __future__ import annotations

import random

import numpy as np


try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
