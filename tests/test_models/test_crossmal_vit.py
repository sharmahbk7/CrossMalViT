"""Tests for CrossMal-ViT model."""

import pytest
import torch

from crossmal_vit.models import CrossMalViT, build_crossmal_vit


@pytest.fixture
def model() -> CrossMalViT:
    return CrossMalViT(
        img_size=224,
        num_classes=18,
        embed_dim=768,
        depth=12,
        num_heads=12,
        pretrained=False,
    )


@pytest.fixture
def sample_batch() -> dict:
    bsz = 2
    return {
        "raw": torch.randn(bsz, 1, 224, 224),
        "entropy": torch.randn(bsz, 1, 224, 224),
        "frequency": torch.randn(bsz, 1, 224, 224),
    }


class TestCrossMalViT:
    def test_forward_shape(self, model: CrossMalViT, sample_batch: dict) -> None:
        outputs = model(sample_batch)
        assert outputs["logits"].shape == (2, 18)

    def test_forward_with_features(self, model: CrossMalViT, sample_batch: dict) -> None:
        outputs = model(sample_batch, return_features=True)
        assert "logits" in outputs
        assert "features" in outputs
        assert "view_features" in outputs
        assert outputs["features"].shape == (2, 2304)

    def test_forward_with_attention(self, model: CrossMalViT, sample_batch: dict) -> None:
        outputs = model(sample_batch, return_attention=True)
        assert "attention" in outputs

    def test_get_cls_tokens(self, model: CrossMalViT, sample_batch: dict) -> None:
        cls_tokens = model.get_cls_tokens(sample_batch)
        assert "raw" in cls_tokens
        assert "entropy" in cls_tokens
        assert "frequency" in cls_tokens
        assert cls_tokens["raw"].shape == (2, 768)

    def test_extract_features(self, model: CrossMalViT, sample_batch: dict) -> None:
        features = model.extract_features(sample_batch)
        assert features.shape == (2, 2304)

    def test_build_from_config(self) -> None:
        config = {
            "img_size": 224,
            "num_classes": 18,
            "embed_dim": 768,
            "pretrained": False,
        }
        model = build_crossmal_vit(config)
        assert isinstance(model, CrossMalViT)

    def test_gradient_flow(self, model: CrossMalViT, sample_batch: dict) -> None:
        outputs = model(sample_batch)
        loss = outputs["logits"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
