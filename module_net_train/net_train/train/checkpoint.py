from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch



def save_checkpoint(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)



def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


@dataclass
class CheckpointManager:
    ckpt_dir: Path
    monitor: str = "val/extent_iou"
    mode: str = "max"
    save_last: bool = True
    save_best: bool = True

    best_value: Optional[float] = None

    def __post_init__(self) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.ckpt_dir / "last.pt"
        self.best_path = self.ckpt_dir / "best.pt"

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def step(self, state: Dict[str, Any], metrics: Dict[str, float]) -> Dict[str, Any]:
        monitor_value = float(metrics.get(self.monitor, float("nan")))
        did_save_best = False

        if self.save_last:
            save_checkpoint(self.last_path, state)

        if self.save_best and monitor_value == monitor_value and self._is_better(monitor_value):
            self.best_value = monitor_value
            save_checkpoint(self.best_path, state)
            did_save_best = True

        return {
            "saved_last": bool(self.save_last),
            "saved_best": did_save_best,
            "best_value": self.best_value,
            "monitor": self.monitor,
            "monitor_value": monitor_value,
            "last_path": str(self.last_path),
            "best_path": str(self.best_path),
        }
