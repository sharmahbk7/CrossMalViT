"""Label-Distribution-Aware Margin Loss for imbalanced classification."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin Loss."""

    def __init__(
        self,
        cls_num_list: list,
        max_margin: float = 0.5,
        weight: Optional[torch.Tensor] = None,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.cls_num_list = cls_num_list
        self.max_margin = max_margin
        self.scale = scale

        cls_num_array = np.array(cls_num_list, dtype=np.float32)
        margins = max_margin / (cls_num_array ** 0.25)
        self.register_buffer("margins", torch.from_numpy(margins).float())

        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_margins = self.margins[targets]
        margin_logits = logits.clone()
        margin_logits[torch.arange(logits.size(0)), targets] -= batch_margins
        margin_logits = margin_logits * self.scale

        loss = F.cross_entropy(margin_logits, targets, weight=self.weight, reduction="mean")
        return loss


class ClassBalancedLDAM(nn.Module):
    """LDAM with class-balanced weighting."""

    def __init__(
        self,
        cls_num_list: list,
        max_margin: float = 0.5,
        beta: float = 0.9999,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        effective_num = 1.0 - np.power(beta, np.array(cls_num_list))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(cls_num_list)

        self.ldam = LDAMLoss(
            cls_num_list=cls_num_list,
            max_margin=max_margin,
            weight=torch.from_numpy(weights).float(),
            scale=scale,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ldam(logits, targets)
