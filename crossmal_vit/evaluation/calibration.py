"""Calibration metrics and visualization."""

from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt


def reliability_diagram(y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> Dict[str, np.ndarray]:
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_count[i] = in_bin.sum()
        if bin_count[i] > 0:
            bin_acc[i] = accuracies[in_bin].mean()
            bin_conf[i] = confidences[in_bin].mean()

    return {"bin_centers": bin_centers, "bin_acc": bin_acc, "bin_conf": bin_conf, "bin_count": bin_count}


def plot_reliability_diagram(
    y_prob: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
    save_path: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    data = reliability_diagram(y_prob, y_true, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.bar(data["bin_centers"], data["bin_acc"], width=1 / n_bins, alpha=0.7, label="Accuracy")
    ax.plot(data["bin_centers"], data["bin_conf"], color="red", marker="o", label="Confidence")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax
