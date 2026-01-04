"""Balanced sampler for imbalanced datasets."""

from typing import Iterable, Optional
import numpy as np
import torch
from torch.utils.data import Sampler, WeightedRandomSampler


class BalancedSampler(Sampler[int]):
    """Weighted sampler that balances classes by inverse frequency."""

    def __init__(
        self,
        labels: Iterable[int],
        num_samples: Optional[int] = None,
        replacement: bool = True,
    ) -> None:
        labels_list = list(labels)
        counts = np.bincount(labels_list)
        weights = 1.0 / np.maximum(counts, 1)
        sample_weights = torch.tensor([weights[label] for label in labels_list], dtype=torch.float)
        self.sampler = WeightedRandomSampler(
            sample_weights, num_samples or len(labels_list), replacement=replacement
        )

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self) -> int:
        return len(self.sampler)
