"""Cross-attention fusion module for multi-view token exchange."""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Cross-attention between two views."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, num_tokens, dim = query.shape

        q = self.q_proj(query).reshape(bsz, num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).reshape(bsz, num_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).reshape(bsz, num_tokens, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(bsz, num_tokens, dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x, None


class CrossAttentionFusion(nn.Module):
    """Multi-view cross-attention fusion module."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 12,
        num_views: int = 3,
        alpha: float = 0.72,
        cross_weights: Optional[Dict[str, float]] = None,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_views = num_views
        self.alpha = alpha

        self.view_names = ["raw", "entropy", "frequency"]

        if cross_weights is None:
            cross_weights = {
                "raw_entropy": 0.34,
                "raw_frequency": 0.28,
                "entropy_frequency": 0.18,
            }
        self.cross_weights = cross_weights

        self.cross_attn = nn.ModuleDict()
        for i, view_a in enumerate(self.view_names):
            for view_b in self.view_names[i + 1 :]:
                self.cross_attn[f"{view_a}_to_{view_b}"] = CrossAttention(
                    dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=drop_rate,
                    proj_drop=drop_rate,
                )
                self.cross_attn[f"{view_b}_to_{view_a}"] = CrossAttention(
                    dim=embed_dim,
                    num_heads=num_heads,
                    attn_drop=drop_rate,
                    proj_drop=drop_rate,
                )

        self.norms = nn.ModuleDict({view: nn.LayerNorm(embed_dim) for view in self.view_names})
        self.fusion_gates = nn.ParameterDict(
            {view: nn.Parameter(torch.ones(1) * alpha) for view in self.view_names}
        )

    def forward(
        self,
        view_tokens: Dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        attention_weights = {} if return_attention else None
        fused_tokens: Dict[str, torch.Tensor] = {}

        for view in self.view_names:
            self_attn_output = view_tokens[view]
            cross_attn_outputs = []

            for other_view in self.view_names:
                if other_view == view:
                    continue
                pair_key = self._get_pair_key(view, other_view)
                weight = self.cross_weights.get(pair_key, 0.2)

                attn_key = f"{other_view}_to_{view}"
                cross_out, attn = self.cross_attn[attn_key](
                    query=view_tokens[view],
                    key_value=view_tokens[other_view],
                    return_attention=return_attention,
                )

                cross_attn_outputs.append(weight * cross_out)

                if return_attention and attn is not None:
                    attention_weights[f"{view}_from_{other_view}"] = attn

            gate = torch.sigmoid(self.fusion_gates[view])
            cross_combined = sum(cross_attn_outputs) if cross_attn_outputs else 0.0
            fused = gate * self_attn_output + (1 - gate) * cross_combined
            fused_tokens[view] = self.norms[view](fused)

        return fused_tokens, attention_weights

    def _get_pair_key(self, view_a: str, view_b: str) -> str:
        views = sorted([view_a, view_b])
        return f"{views[0]}_{views[1]}"
