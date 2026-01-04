"""Local entropy computation for malware byteplots."""

import numpy as np
from scipy.ndimage import uniform_filter


def compute_local_entropy(
    image: np.ndarray,
    window_size: int = 9,
    num_bins: int = 256,
) -> np.ndarray:
    """Compute local Shannon entropy using a sliding window."""
    image = image.astype(np.float64)

    if image.max() > 0:
        quantized = (image / (image.max() + 1e-8) * (num_bins - 1)).astype(np.int32)
    else:
        quantized = np.zeros_like(image, dtype=np.int32)

    height, width = image.shape
    entropy_map = np.zeros((height, width), dtype=np.float64)

    pad = window_size // 2
    padded = np.pad(quantized, pad, mode="reflect")

    for i in range(height):
        for j in range(width):
            window = padded[i : i + window_size, j : j + window_size].flatten()
            hist, _ = np.histogram(window, bins=num_bins, range=(0, num_bins - 1))
            hist = hist.astype(np.float64)

            total = hist.sum()
            if total > 0:
                prob = hist / total
                prob = prob[prob > 0]
                entropy_map[i, j] = -np.sum(prob * np.log2(prob))

    return entropy_map


def compute_local_entropy_fast(image: np.ndarray, window_size: int = 9) -> np.ndarray:
    """Fast approximation of local entropy using local variance."""
    image = image.astype(np.float64)

    local_mean = uniform_filter(image, size=window_size, mode="reflect")
    local_sq_mean = uniform_filter(image ** 2, size=window_size, mode="reflect")
    local_var = local_sq_mean - local_mean ** 2
    local_var = np.maximum(local_var, 1e-10)

    entropy_map = 0.5 * np.log2(2 * np.pi * np.e * local_var)
    entropy_map = np.clip(entropy_map, 0, 8)
    return entropy_map
