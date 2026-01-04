"""Fusion utilities for multi-view classification."""

from typing import Dict, List
import torch
import torch.nn as nn
from .classification_head import ClassificationHead


class LateFusion(nn.Module):
    """Late fusion classifier using per-view heads."""

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 18,
        view_names: List[str] = None,
        fusion: str = "average",
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.view_names = view_names or ["raw", "entropy", "frequency"]
        self.fusion = fusion

        self.heads = nn.ModuleDict(
            {
                view: ClassificationHead(
                    in_features=embed_dim,
                    hidden_features=embed_dim,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                )
                for view in self.view_names
            }
        )

        if fusion == "learned":
            self.fusion_weights = nn.Parameter(torch.ones(len(self.view_names)))
        else:
            self.fusion_weights = None

    def forward(self, view_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        logits = []
        for view in self.view_names:
            logits.append(self.heads[view](view_features[view]))
        stacked = torch.stack(logits, dim=1)

        if self.fusion == "average":
            return stacked.mean(dim=1)
        if self.fusion == "max":
            return stacked.max(dim=1).values
        if self.fusion == "learned":
            weights = torch.softmax(self.fusion_weights, dim=0)
            return (stacked * weights.view(1, -1, 1)).sum(dim=1)

        raise ValueError(f"Unsupported fusion strategy: {self.fusion}")
