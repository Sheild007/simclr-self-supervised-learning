"""Visualization helpers for Assignment 5."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay

from utils.metrics import CIFAR10_CLASSES


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_loss_curves(train_losses: Sequence[float],val_losses: Sequence[float],out_path: str | Path,title: str = "Training and validation loss",val_accuracies: Sequence[float] | None = None ) -> None:
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_losses, label="Train loss", marker="o", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val loss", marker="s", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if val_accuracies is not None and len(val_accuracies) == len(train_losses):
        ax2 = ax.twinx()
        ax2.plot(
            epochs,
            np.asarray(val_accuracies) * 100.0,
            label="Val accuracy (%)",
            color="tab:green",
            linestyle="--",
            marker="^",
            linewidth=1.2,
        )
        ax2.set_ylabel("Val accuracy (%)")
        ax2.set_ylim(0, 100)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    else:
        ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray,out_path: str | Path,title: str = "Confusion Matrix",class_names: Sequence[str] | None = None,normalize: bool = False,) -> None:
 
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if class_names is None:
        class_names = CIFAR10_CLASSES

    if normalize:
        matrix = cm.astype(np.float64)
        row_sums = matrix.sum(axis=1, keepdims=True)
        matrix = np.divide(matrix, row_sums, where=row_sums > 0)
        values_format = ".2f"
    else:
        matrix = cm.astype(np.int64)
        values_format = "d"

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=list(class_names),
    )
    disp.plot(
        ax=ax,
        xticks_rotation=45,
        colorbar=True,
        values_format=values_format,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def denormalize_cifar10(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=tensor.device).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


def plot_augmentation_examples(originals,view1s,view2s,save_path: str | Path,n_rows: int = 10 ) -> None:
    
    save_augmentation_grid(originals, view1s, view2s, out_path=save_path, max_rows=n_rows)


def plot_similarity_heatmap(matrix,out_path: str | Path,title: str = "Cosine similarity",batch_size: int | None = None,) -> None:
    
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()
    matrix = np.asarray(matrix)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_title(title)
    ax.set_xlabel("view index")
    ax.set_ylabel("view index")
    fig.colorbar(im, ax=ax, label="cosine similarity")

    if batch_size is not None:
        n = int(batch_size)
        two_n = matrix.shape[0]
        if two_n == 2 * n:
            from matplotlib.patches import Rectangle
            # Positive-pair blocks: top-right and bottom-left N x N diagonals.
            ax.add_patch(Rectangle((n - 0.5, -0.5), n, n,
                                   fill=False, edgecolor="black",
                                   linewidth=0.8, linestyle="--"))
            ax.add_patch(Rectangle((-0.5, n - 0.5), n, n,
                                   fill=False, edgecolor="black",
                                   linewidth=0.8, linestyle="--"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_augmentation_grid(originals, view1s, view2s, out_path: str | Path, max_rows: int = 10) -> None:
    """Save a grid: Original | View 1 | View 2."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = min(max_rows, len(originals))
    fig, axes = plt.subplots(rows, 3, figsize=(6, 2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for r in range(rows):
        imgs = [originals[r], view1s[r], view2s[r]]
        titles = ["Original", "View 1", "View 2"]
        for c in range(3):
            img = imgs[c]
            if isinstance(img, torch.Tensor):
                if img.ndim == 3 and img.shape[0] == 3:
                    img = denormalize_cifar10(img.detach().cpu()).permute(1, 2, 0).numpy()
            axes[r, c].imshow(img)
            axes[r, c].set_title(titles[c])
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_2d_feature_plot(
    features: np.ndarray,
    labels: np.ndarray,
    out_path: str | Path,
    method: str = "pca",
    title: str = "Feature Visualization",
    seed: int = 2026,
) -> None:
    """Save PCA or t-SNE visualization of features.

    Labels should be used only for coloring the plot, not for SSL training.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    method = method.lower()
    if method == "pca":
        coords = PCA(n_components=2, random_state=seed).fit_transform(features)
    elif method in {"tsne", "t-sne"}:
        coords = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30, random_state=seed).fit_transform(features)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, alpha=0.75)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, ticks=range(10))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
