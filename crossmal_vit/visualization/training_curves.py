"""Training curve visualization utilities."""

from typing import Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd


def load_lightning_metrics(csv_path: str) -> Dict[str, list]:
    df = pd.read_csv(csv_path)
    history = {}
    for column in df.columns:
        if column == "step":
            continue
        history[column] = df[column].dropna().tolist()
    return history


def plot_training_curves(history: Dict[str, list], save_path: str = None) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, values in history.items():
        if name.lower().startswith("val/") or name.lower().startswith("train/"):
            ax.plot(values, label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric")
    ax.set_title("Training Curves")
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax
