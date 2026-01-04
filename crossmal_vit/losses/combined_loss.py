"""Combined classification and contrastive loss."""

from typing import Dict
import torch
import torch.nn as nn


class CombinedLoss(nn.Module):
    """Combined classification and contrastive loss."""

    def __init__(
        self,
        classification_loss: nn.Module,
        contrastive_loss: nn.Module,
        lambda_contrast: float = 0.32,
    ) -> None:
        super().__init__()
        self.cls_loss = classification_loss
        self.contrast_loss = contrastive_loss
        self.lambda_contrast = lambda_contrast

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        view_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cls_loss = self.cls_loss(logits, targets)
        contrast_loss = self.contrast_loss(view_features)
        total_loss = cls_loss + self.lambda_contrast * contrast_loss

        return {
            "total": total_loss,
            "classification": cls_loss,
            "contrastive": contrast_loss,
        }
