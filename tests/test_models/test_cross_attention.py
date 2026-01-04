"""Tests for cross-attention fusion."""

import torch

from crossmal_vit.models.cross_attention import CrossAttentionFusion


def test_cross_attention_fusion_shapes() -> None:
    fusion = CrossAttentionFusion(embed_dim=64, num_heads=4)
    bsz, num_tokens, dim = 2, 8, 64
    views = {
        "raw": torch.randn(bsz, num_tokens, dim),
        "entropy": torch.randn(bsz, num_tokens, dim),
        "frequency": torch.randn(bsz, num_tokens, dim),
    }
    fused, attn = fusion(views, return_attention=True)
    assert fused["raw"].shape == (bsz, num_tokens, dim)
    assert attn is not None
