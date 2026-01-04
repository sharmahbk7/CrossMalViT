"""Tests for data transforms."""

import numpy as np
import torch

from crossmal_vit.data.transforms import MultiViewTransform, compute_local_entropy, compute_frequency_energy


def test_multi_view_transform_shapes() -> None:
    image = np.random.randint(0, 256, size=(224, 224), dtype=np.uint8)
    transform = MultiViewTransform(img_size=224, entropy_window=9)
    views = transform(image)
    assert set(views.keys()) == {"raw", "entropy", "frequency"}
    for view in views.values():
        assert view.shape == (1, 224, 224)
        assert view.dtype == torch.float32


def test_entropy_map_range() -> None:
    image = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    entropy = compute_local_entropy(image, window_size=5)
    assert entropy.min() >= 0


def test_frequency_map_range() -> None:
    image = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    freq = compute_frequency_energy(image)
    assert freq.min() >= 0
    assert freq.max() <= 1.0
