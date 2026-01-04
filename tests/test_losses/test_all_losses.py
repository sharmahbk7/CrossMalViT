"""Tests for loss functions."""

import torch

from crossmal_vit.losses import FocalLoss, ClassBalancedLoss, LDAMLoss, MultiViewContrastiveLoss


def test_focal_loss() -> None:
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    loss_fn = FocalLoss(gamma=2.0)
    loss = loss_fn(logits, targets)
    assert loss.item() > 0


def test_class_balanced_loss() -> None:
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    loss_fn = ClassBalancedLoss([10, 5, 2])
    loss = loss_fn(logits, targets)
    assert loss.item() > 0


def test_ldam_loss() -> None:
    logits = torch.randn(4, 3)
    targets = torch.tensor([0, 1, 2, 1])
    loss_fn = LDAMLoss([10, 5, 2])
    loss = loss_fn(logits, targets)
    assert loss.item() > 0


def test_contrastive_loss() -> None:
    features = {
        "raw": torch.randn(4, 768),
        "entropy": torch.randn(4, 768),
        "frequency": torch.randn(4, 768),
    }
    loss_fn = MultiViewContrastiveLoss(embed_dim=768)
    loss = loss_fn(features)
    assert loss.item() > 0
