from __future__ import annotations

import pytest

from net_train.utils.seed import make_torch_generator, seed_dataloader_worker

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def test_seed_dataloader_worker_smoke() -> None:
    # Outside real DataLoader workers this should still be safe.
    seed_dataloader_worker(0)


def test_make_torch_generator_reproducible() -> None:
    if torch is None:
        pytest.skip("torch is not available")

    g1 = make_torch_generator(123)
    g2 = make_torch_generator(123)
    assert g1 is not None
    assert g2 is not None

    a = torch.rand((8,), generator=g1)
    b = torch.rand((8,), generator=g2)
    assert torch.allclose(a, b)
