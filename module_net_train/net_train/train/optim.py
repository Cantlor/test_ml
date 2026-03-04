from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR



def create_optimizer(train_cfg: Dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    opt_cfg = ((train_cfg.get("train", {}) or {}).get("optimizer", {}) or {})
    name = str(opt_cfg.get("name", "adamw")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 1e-4))

    if name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    raise ValueError(f"Unknown optimizer: {name}")



def _cosine_with_warmup_lambda(epoch: int, warmup_epochs: int, total_epochs: int, min_lr_ratio: float) -> float:
    if total_epochs <= 0:
        return 1.0

    if epoch < warmup_epochs:
        return float(epoch + 1) / float(max(1, warmup_epochs))

    progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
    cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)



def create_scheduler(train_cfg: Dict[str, Any], optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = ((train_cfg.get("train", {}) or {}).get("scheduler", {}) or {})
    name = str(sched_cfg.get("name", "cosine")).lower()

    if name in {"none", "off", "null"}:
        return None

    if name == "cosine":
        epochs = int(((train_cfg.get("train", {}) or {}).get("epochs", 1)))
        warmup = int(sched_cfg.get("warmup_epochs", 0))
        min_lr = float(sched_cfg.get("min_lr", 1e-5))
        base_lr = float((((train_cfg.get("train", {}) or {}).get("optimizer", {}) or {}).get("lr", 1e-3)))
        min_lr_ratio = float(min_lr / max(base_lr, 1e-12))

        def _fn(ep: int) -> float:
            return _cosine_with_warmup_lambda(
                epoch=ep,
                warmup_epochs=warmup,
                total_epochs=epochs,
                min_lr_ratio=min_lr_ratio,
            )

        return LambdaLR(optimizer, lr_lambda=_fn)

    raise ValueError(f"Unknown scheduler: {name}")
