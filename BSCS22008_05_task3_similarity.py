from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch

from utils.seed import set_seed
from utils.dataset_splits import (
    get_cifar10_train,
    load_indices,
    IndexedSubset,
    build_loader,
)
from utils.augmentations import build_simclr_transform, TwoViewTransform
from utils.models import CIFARResNet18
from utils.losses import (
    cosine_similarity_matrix,
    mean_same_image_similarity,
    mean_different_image_similarity,
)
from utils.visualization import plot_similarity_heatmap, ensure_dir


SMALL_BATCH_FOR_HEATMAP: int = 8   # gives a 16x16 heatmap (easy to read)
EVAL_BATCH: int = 64               # larger batch for stable similarity averages

HEATMAP_PATH: str = "results/similarity_matrix_before_training.png"
METRICS_PATH: str = "results/metrics_similarity_before.json"


def compute_similarities_with_random_encoder(
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    """Run an UNTRAINED encoder on one batch of ``batch_size`` val images.

    Returns a dict with the two summary numbers plus the (2N, 512) feature
    tensor so the heatmap can be plotted without recomputing.
    """
    simclr_t = build_simclr_transform()
    two_view = TwoViewTransform(simclr_t)

    indices = load_indices("val")[:batch_size]
    dataset = IndexedSubset(get_cifar10_train(), indices, transform=two_view)
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False)

    encoder = CIFARResNet18().to(device).eval()

    (views, _) = next(iter(loader))
    v1, v2 = views
    x = torch.cat([v1, v2], dim=0).to(device)
    with torch.no_grad():
        h = encoder(x)

    same = mean_same_image_similarity(h, batch_size)
    diff = mean_different_image_similarity(h, batch_size)
    return {
        "same_view": same,
        "different_image": diff,
        "features": h.detach().cpu(),
        "batch_size": batch_size,
    }


def save_before_heatmap(features: torch.Tensor, batch_size: int) -> None:
    """Plot the 2N x 2N cosine similarity matrix for the untrained encoder."""
    S = cosine_similarity_matrix(features)
    plot_similarity_heatmap(
        S,
        out_path=HEATMAP_PATH,
        title=f"Cosine similarity (untrained encoder, N={batch_size})",
        batch_size=batch_size,
    )


def main() -> None:
    ensure_dir("results")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task3] device={device}")

    small = compute_similarities_with_random_encoder(SMALL_BATCH_FOR_HEATMAP, device)
    save_before_heatmap(small["features"], small["batch_size"])
    print(f"[task3] saved {HEATMAP_PATH}")

    big = compute_similarities_with_random_encoder(EVAL_BATCH, device)
    same = float(big["same_view"])
    diff = float(big["different_image"])
    print(f"[task3] same_view_similarity_before:       {same:.4f}")
    print(f"[task3] different_image_similarity_before: {diff:.4f}")
    print(f"[task3] gap (same - diff):                {same - diff:+.4f}")

    metrics = {
        "same_view_similarity_before": same,
        "different_image_similarity_before": diff,
        "gap_before": same - diff,
        "eval_batch_size": EVAL_BATCH,
        "small_batch_for_heatmap": SMALL_BATCH_FOR_HEATMAP,
        "encoder": "CIFARResNet18 (random init, untrained)",
        "split": "val (first {} images)".format(EVAL_BATCH),
    }
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2))
    print(f"[task3] saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
