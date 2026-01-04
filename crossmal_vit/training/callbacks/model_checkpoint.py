"""Model checkpointing utility."""

from pathlib import Path
from typing import Optional
import torch


class BestModelCheckpoint:
    """Save the best model based on a monitored metric."""

    def __init__(self, monitor: str = "accuracy", mode: str = "max") -> None:
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def _is_better(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return current > self.best
        return current < self.best

    def step(self, metrics: dict, model: torch.nn.Module, path: str, epoch: int) -> Optional[str]:
        current = float(metrics.get(self.monitor, 0.0))
        if self._is_better(current):
            self.best = current
            path_obj = Path(path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)
            save_path = path_obj.with_name(f"best_epoch_{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            return str(save_path)
        return None
