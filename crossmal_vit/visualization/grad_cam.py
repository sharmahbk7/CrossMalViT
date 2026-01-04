"""Gradient-based attribution for CrossMal-ViT inputs."""

from typing import Dict, Optional
import torch
import numpy as np


def compute_input_grad_cam(
    model: torch.nn.Module,
    views: Dict[str, torch.Tensor],
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Compute a simple gradient-based attribution on the raw view."""
    model.eval()
    raw = views["raw"].clone().requires_grad_(True)
    views = {**views, "raw": raw}

    outputs = model(views)
    logits = outputs["logits"]

    if target_class is None:
        target_class = int(logits.argmax(dim=-1)[0].item())

    score = logits[:, target_class].sum()
    model.zero_grad()
    score.backward()

    grad = raw.grad.detach().abs().mean(dim=1)
    grad = grad[0].cpu().numpy()
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    return grad
