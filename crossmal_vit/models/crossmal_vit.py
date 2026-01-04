"""CrossMal-ViT: Main model integrating three-view ViT encoders."""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from .vit_encoder import ViTEncoder
from .cross_attention import CrossAttentionFusion
from .classification_head import ClassificationHead


class CrossMalViT(nn.Module):
    """Multi-view Vision Transformer for malware classification."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        num_classes: int = 18,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        fusion_layers: List[int] = None,
        fusion_alpha: float = 0.72,
        cross_weights: Optional[Dict[str, float]] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.fusion_layers = fusion_layers or [4, 7, 10]
        self.view_names = ["raw", "entropy", "frequency"]

        if cross_weights is None:
            cross_weights = {
                "raw_entropy": 0.34,
                "raw_frequency": 0.28,
                "entropy_frequency": 0.18,
            }
        self.cross_weights = cross_weights

        self.encoders = nn.ModuleDict(
            {
                view: ViTEncoder(
                    img_size=img_size,
                    patch_size=patch_size,
                    in_chans=in_chans,
                    embed_dim=embed_dim,
                    depth=depth,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    pretrained=pretrained,
                )
                for view in self.view_names
            }
        )

        self.fusion_modules = nn.ModuleDict(
            {
                f"layer_{layer}": CrossAttentionFusion(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_views=3,
                    alpha=fusion_alpha,
                    cross_weights=cross_weights,
                    drop_rate=drop_rate,
                )
                for layer in self.fusion_layers
            }
        )

        self.head = ClassificationHead(
            in_features=embed_dim * 3,
            hidden_features=1024,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )

        self.attention_maps: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        return_features: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        view_tokens: Dict[str, torch.Tensor] = {}
        for view in self.view_names:
            view_tokens[view] = self.encoders[view].patch_embed(x[view])
            view_tokens[view] = self.encoders[view].prepare_tokens(view_tokens[view])

        for layer_idx in range(self.encoders["raw"].depth):
            for view in self.view_names:
                view_tokens[view] = self.encoders[view].blocks[layer_idx](view_tokens[view])

            if layer_idx in self.fusion_layers:
                fusion_key = f"layer_{layer_idx}"
                view_tokens, attn_weights = self.fusion_modules[fusion_key](
                    view_tokens, return_attention=return_attention
                )
                if return_attention and attn_weights is not None:
                    self.attention_maps[fusion_key] = attn_weights

        for view in self.view_names:
            view_tokens[view] = self.encoders[view].norm(view_tokens[view])

        cls_tokens = torch.cat([view_tokens[view][:, 0] for view in self.view_names], dim=-1)
        logits = self.head(cls_tokens)

        output: Dict[str, torch.Tensor] = {"logits": logits}
        if return_features:
            output["features"] = cls_tokens
            output["view_features"] = {view: view_tokens[view][:, 0] for view in self.view_names}
        if return_attention:
            output["attention"] = self.attention_maps

        return output

    def get_cls_tokens(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self.forward(x, return_features=True)
        return output["view_features"]

    @torch.no_grad()
    def extract_features(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.forward(x, return_features=True)
        return output["features"]


def build_crossmal_vit(config: Dict) -> CrossMalViT:
    """Factory function to build CrossMal-ViT from config."""
    return CrossMalViT(
        img_size=config.get("img_size", 224),
        patch_size=config.get("patch_size", 16),
        in_chans=config.get("in_chans", 1),
        num_classes=config.get("num_classes", 18),
        embed_dim=config.get("embed_dim", 768),
        depth=config.get("depth", 12),
        num_heads=config.get("num_heads", 12),
        mlp_ratio=config.get("mlp_ratio", 4.0),
        fusion_layers=config.get("fusion_layers", [4, 7, 10]),
        fusion_alpha=config.get("fusion_alpha", 0.72),
        cross_weights=config.get("cross_weights", None),
        drop_rate=config.get("drop_rate", 0.0),
        attn_drop_rate=config.get("attn_drop_rate", 0.0),
        drop_path_rate=config.get("drop_path_rate", 0.1),
        pretrained=config.get("pretrained", True),
    )
