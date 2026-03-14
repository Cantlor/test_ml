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


def make_torch_generator(seed: int):
    if torch is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def seed_dataloader_worker(worker_id: int) -> None:
    # torch initial_seed() is different per worker when DataLoader gets `generator`.
    if torch is not None:
        worker_seed = int(torch.initial_seed() % (2 ** 32))
    else:
        worker_seed = int(worker_id)

    random.seed(worker_seed)
    np.random.seed(worker_seed)

    if torch is not None:
        torch.manual_seed(worker_seed)
        try:
            info = torch.utils.data.get_worker_info()
        except Exception:
            info = None
        if info is not None and info.dataset is not None:
            ds = info.dataset
            if hasattr(ds, "rng"):
                ds.rng = np.random.default_rng(worker_seed)
            aug = getattr(ds, "augmentor", None)
            if aug is not None and hasattr(aug, "rng"):
                aug.rng = np.random.default_rng(worker_seed + 1)
