"""Attention map visualization for CrossMal-ViT."""

from typing import Dict
import numpy as np
import torch
import matplotlib.pyplot as plt


def extract_attention_maps(
    model: torch.nn.Module,
    views: Dict[str, torch.Tensor],
    layer_key: str = "layer_4",
) -> Dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        outputs = model(views, return_attention=True)
    attention = outputs.get("attention", {})
    return attention.get(layer_key, {}) if attention else {}


def save_attention_overlays(
    image: np.ndarray,
    attention_maps: Dict[str, torch.Tensor],
    save_dir: str,
    alpha: float = 0.6,
) -> None:
    import os

    os.makedirs(save_dir, exist_ok=True)

    for key, attn in attention_maps.items():
        if attn.ndim == 4:
            attn_map = attn.mean(dim=1).mean(dim=1).cpu().numpy()
        else:
            attn_map = attn.cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)

        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap="gray")
        plt.imshow(attn_map, cmap="jet", alpha=alpha)
        plt.axis("off")
        plt.title(key)
        plt.savefig(os.path.join(save_dir, f"{key}.png"), bbox_inches="tight")
        plt.close()
