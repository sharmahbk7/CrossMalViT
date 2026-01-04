"""Early stopping utility."""

from typing import Optional


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: Optional[float] = None
        self.num_bad_epochs = 0

    def step(self, current: float) -> bool:
        if self.best is None:
            self.best = current
            return False

        improved = (
            (current - self.best) > self.min_delta
            if self.mode == "max"
            else (self.best - current) > self.min_delta
        )

        if improved:
            self.best = current
            self.num_bad_epochs = 0
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience
