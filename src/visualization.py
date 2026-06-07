"""Visualisations : comparaison des metriques et projection des embeddings.

Genere des graphiques PNG dans results/ :
  - comparison_metrics.png : barres groupees P@10, Recall@10, MRR, NDCG@10
  - comparison_time.png    : temps moyen de recherche par methode
  - embeddings_projection.png : projection 2D (UMAP si dispo, sinon t-SNE)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # backend non-interactif pour la generation de fichiers
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config

sns.set_theme(style="whitegrid")


def plot_metric_comparison(
    summary: Dict[str, Dict],
    metrics: Optional[List[str]] = None,
    save_path=None,
):
    """Barres groupees comparant les methodes sur plusieurs metriques."""
    metrics = metrics or ["P@10", "Recall@10", "MRR", "NDCG@10"]
    methods = list(summary.keys())

    x = np.arange(len(metrics))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, method in enumerate(methods):
        values = [summary[method].get(m, 0.0) for m in metrics]
        ax.bar(x + i * width, values, width, label=method)

    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des methodes de recherche")
    ax.legend(title="Methode")
    fig.tight_layout()

    save_path = save_path or (config.RESULTS_DIR / "comparison_metrics.png")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_time_comparison(summary: Dict[str, Dict], save_path=None):
    """Barres du temps moyen de recherche par methode."""
    methods = list(summary.keys())
    times = [summary[m].get("time_mean_ms", 0.0) for m in methods]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, times, color=sns.color_palette("viridis", len(methods)))
    ax.set_ylabel("Temps moyen (ms)")
    ax.set_title("Temps moyen de recherche par methode")
    ax.bar_label(bars, fmt="%.1f")
    fig.tight_layout()

    save_path = save_path or (config.RESULTS_DIR / "comparison_time.png")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    method: str = "auto",
    max_points: int = 2000,
    save_path=None,
):
    """Projette les embeddings en 2D (UMAP si installe, sinon t-SNE)."""
    n = embeddings.shape[0]
    if n > max_points:
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = rng.choice(n, size=max_points, replace=False)
        emb = embeddings[idx]
    else:
        emb = embeddings

    coords = None
    used = method
    if method in ("auto", "umap"):
        try:
            import umap

            reducer = umap.UMAP(random_state=config.RANDOM_SEED)
            coords = reducer.fit_transform(emb)
            used = "UMAP"
        except Exception:
            coords = None

    if coords is None:
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2, random_state=config.RANDOM_SEED, init="pca",
            perplexity=min(30, max(5, emb.shape[0] // 4)),
        )
        coords = reducer.fit_transform(emb)
        used = "t-SNE"

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords[:, 0], coords[:, 1], s=8, alpha=0.5)
    ax.set_title(f"Projection 2D des embeddings d'emails ({used})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()

    save_path = save_path or (config.RESULTS_DIR / "embeddings_projection.png")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
