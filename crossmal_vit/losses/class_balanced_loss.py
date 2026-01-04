"""Class-balanced loss implementations."""

from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .focal_loss import FocalLoss


def compute_class_balanced_weights(cls_num_list: List[int], beta: float = 0.9999) -> torch.Tensor:
    """Compute class-balanced weights using effective number of samples."""
    effective_num = 1.0 - np.power(beta, np.array(cls_num_list))
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * len(cls_num_list)
    return torch.tensor(weights, dtype=torch.float32)


class ClassBalancedLoss(nn.Module):
    """Class-balanced loss with optional focal modulation."""

    def __init__(
        self,
        cls_num_list: List[int],
        loss_type: str = "focal",
        beta: float = 0.9999,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weights = compute_class_balanced_weights(cls_num_list, beta=beta)
        self.loss_type = loss_type
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "focal":
            focal = FocalLoss(gamma=self.gamma, alpha=self.weights, reduction=self.reduction)
            return focal(logits, targets)
        return F.cross_entropy(logits, targets, weight=self.weights.to(logits.device))
