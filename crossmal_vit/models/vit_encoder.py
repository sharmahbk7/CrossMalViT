"""Vision Transformer encoder for single-view processing."""

import torch
import torch.nn as nn
from functools import partial
from timm.layers import trunc_normal_
from .components.patch_embed import PatchEmbed
from .components.attention import Attention
from .components.mlp import Mlp
from .components.drop_path import DropPath


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTEncoder(nn.Module):
    """Vision Transformer encoder for single-view processing."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

        if pretrained:
            self._load_pretrained()

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _load_pretrained(self) -> None:
        """Load pretrained ViT weights and adapt for single-channel input."""
        try:
            import timm

            pretrained_model = timm.create_model("vit_base_patch16_224", pretrained=True)
            state_dict = pretrained_model.state_dict()

            pretrained_patch = state_dict["patch_embed.proj.weight"]
            adapted_patch = pretrained_patch.mean(dim=1, keepdim=True)
            state_dict["patch_embed.proj.weight"] = adapted_patch

            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as exc:
            print(f"Could not load pretrained weights: {exc}")

    def prepare_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Add CLS token and position embeddings."""
        bsz = x.shape[0]
        cls_tokens = self.cls_token.expand(bsz, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.prepare_tokens(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return x[:, 0]
