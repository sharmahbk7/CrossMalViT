"""Multi-view contrastive loss for view alignment."""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewContrastiveLoss(nn.Module):
    """NT-Xent style contrastive loss for multi-view learning."""

    def __init__(
        self,
        embed_dim: int = 768,
        proj_dim: int = 256,
        hidden_dim: int = 512,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature

        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, view_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        view_names = list(view_features.keys())
        projections = {}
        for view_name, features in view_features.items():
            z = self.projector(features)
            z = F.normalize(z, dim=-1)
            projections[view_name] = z

        total_loss = 0.0
        num_pairs = 0

        for i, view_a in enumerate(view_names):
            for view_b in view_names[i + 1 :]:
                loss_ab = self._contrastive_loss(projections[view_a], projections[view_b])
                total_loss += loss_ab
                num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else total_loss

    def _contrastive_loss(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        bsz = z_a.shape[0]
        device = z_a.device
        sim = torch.matmul(z_a, z_b.T) / self.temperature
        labels = torch.arange(bsz, device=device)

        loss_a = F.cross_entropy(sim, labels)
        loss_b = F.cross_entropy(sim.T, labels)
        return (loss_a + loss_b) / 2
