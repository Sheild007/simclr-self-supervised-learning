from __future__ import annotations

import json
from pathlib import Path

import torch

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_train, load_indices, IndexedSubset, build_loader
from utils.augmentations import build_simclr_transform, TwoViewTransform
from utils.models import CIFARResNet18
from utils.losses import cosine_similarity_matrix, mean_same_image_similarity, mean_different_image_similarity
from utils.visualization import plot_similarity_heatmap, ensure_dir


SMALL_BATCH = 8
EVAL_BATCH = 64
HEATMAP_PATH = "results/similarity_matrix_before_training.png"
METRICS_PATH = "results/metrics_similarity_before.json"


def compute_similarities_with_random_encoder(batch_size, device):
    two_view = TwoViewTransform(build_simclr_transform())
    indices = load_indices("val")[:batch_size]
    dataset = IndexedSubset(get_cifar10_train(), indices, transform=two_view)
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False)

    encoder = CIFARResNet18().to(device).eval()
    (views, _) = next(iter(loader))
    v1, v2 = views
    x = torch.cat([v1, v2], dim=0).to(device)
    with torch.no_grad():
        h = encoder(x)

    return {
        "same_view": mean_same_image_similarity(h, batch_size),
        "different_image": mean_different_image_similarity(h, batch_size),
        "features": h.detach().cpu(),
        "batch_size": batch_size,
    }


def save_before_heatmap(features, batch_size):
    S = cosine_similarity_matrix(features)
    plot_similarity_heatmap(S, out_path=HEATMAP_PATH,
                            title=f"Cosine similarity (untrained encoder, N={batch_size})",
                            batch_size=batch_size)


def main():
    ensure_dir("results")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    small = compute_similarities_with_random_encoder(SMALL_BATCH, device)
    save_before_heatmap(small["features"], small["batch_size"])
    print(f"saved {HEATMAP_PATH}")

    big = compute_similarities_with_random_encoder(EVAL_BATCH, device)
    same = float(big["same_view"])
    diff = float(big["different_image"])
    print(f"same view  {same:.4f}")
    print(f"diff image {diff:.4f}")
    print(f"gap        {same - diff:+.4f}")

    out = {
        "same_view_similarity_before": same,
        "different_image_similarity_before": diff,
        "gap_before": same - diff,
        "eval_batch_size": EVAL_BATCH,
        "small_batch_for_heatmap": SMALL_BATCH,
        "encoder": "CIFARResNet18 (random init, untrained)",
        "split": f"val (first {EVAL_BATCH} images)",
    }
    Path(METRICS_PATH).write_text(json.dumps(out, indent=2))
    print(f"saved {METRICS_PATH}")


if __name__ == "__main__":
    main()
