"""Frequency domain energy map computation."""

import numpy as np
from scipy.fft import fft2, fftshift


def compute_frequency_energy(
    image: np.ndarray,
    log_scale: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """Compute frequency-domain energy map using 2D FFT."""
    image = image.astype(np.float64)
    height, width = image.shape
    window_h = np.hanning(height)
    window_w = np.hanning(width)
    window_2d = np.outer(window_h, window_w)
    windowed = image * window_2d

    fft_result = fft2(windowed)
    fft_shifted = fftshift(fft_result)
    magnitude = np.abs(fft_shifted)

    if log_scale:
        magnitude = np.log1p(magnitude)

    if normalize:
        min_val = magnitude.min()
        max_val = magnitude.max()
        if max_val - min_val > 1e-10:
            magnitude = (magnitude - min_val) / (max_val - min_val)
        else:
            magnitude = np.zeros_like(magnitude)

    return magnitude.astype(np.float32)


def compute_frequency_bands(image: np.ndarray, num_bands: int = 4) -> np.ndarray:
    """Compute energy in different frequency bands."""
    height, width = image.shape
    fft_result = fft2(image.astype(np.float64))
    fft_shifted = fftshift(fft_result)
    magnitude = np.abs(fft_shifted) ** 2

    cy, cx = height // 2, width // 2
    y, x = np.ogrid[:height, :width]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)

    band_energies = np.zeros(num_bands)
    band_edges = np.linspace(0, max_r, num_bands + 1)

    for i in range(num_bands):
        mask = (r >= band_edges[i]) & (r < band_edges[i + 1])
        band_energies[i] = magnitude[mask].sum()

    total = band_energies.sum()
    if total > 0:
        band_energies /= total

    return band_energies
