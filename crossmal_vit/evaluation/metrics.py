"""Evaluation metrics for malware classification."""

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def compute_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[list] = None,
) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    metrics["per_class_f1"] = per_class_f1.tolist()

    unique, counts = np.unique(y_true, return_counts=True)
    minority_classes = unique[np.argsort(counts)[:5]]
    minority_mask = np.isin(y_true, minority_classes)
    if minority_mask.sum() > 0:
        metrics["minority_f1"] = f1_score(
            y_true[minority_mask],
            y_pred[minority_mask],
            average="macro",
            zero_division=0,
        )

    return metrics


def compute_ece(y_prob: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            avg_accuracy = accuracies[in_bin].mean()
            avg_confidence = confidences[in_bin].mean()
            ece += (in_bin.sum() / len(y_true)) * abs(avg_accuracy - avg_confidence)
    return float(ece)


def compute_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, normalize: str = "true") -> np.ndarray:
    return confusion_matrix(y_true, y_pred, normalize=normalize)
