"""t-SNE visualization for learned features."""

from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def compute_tsne(features: np.ndarray, labels: np.ndarray, perplexity: float = 30.0) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    return tsne.fit_transform(features)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="tab20", s=8)

    if class_names:
        handles, _ = scatter.legend_elements(prop="colors")
        ax.legend(handles, class_names, title="Classes", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_title("t-SNE of Features")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax
