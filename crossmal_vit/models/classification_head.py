"""Classification head for CrossMal-ViT."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """MLP classification head."""

    def __init__(
        self,
        in_features: int = 2304,
        hidden_features: int = 1024,
        num_classes: int = 18,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(hidden_features, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return x
