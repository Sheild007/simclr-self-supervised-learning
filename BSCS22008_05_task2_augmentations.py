from __future__ import annotations

from typing import List, Tuple

import torch
import torchvision.transforms as T

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_train, load_indices
from utils.augmentations import build_simclr_transform, TwoViewTransform
from utils.visualization import plot_augmentation_examples, ensure_dir


NUM_EXAMPLES: int = 10  # spec: at least 10
OUT_PATH: str = "results/augmentation_examples.png"


def collect_examples(num_examples: int = NUM_EXAMPLES,) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
  
    cifar = get_cifar10_train()  # PIL images, no transform applied
    indices = load_indices("train_ssl_unlabeled")[:num_examples]

    simclr_t = build_simclr_transform()
    two_view = TwoViewTransform(simclr_t)
    to_tensor = T.ToTensor()

    originals: List[torch.Tensor] = []
    views1: List[torch.Tensor] = []
    views2: List[torch.Tensor] = []
    for idx in indices:
        pil_img, _ = cifar[idx]
        v1, v2 = two_view(pil_img)
        originals.append(to_tensor(pil_img))
        views1.append(v1)
        views2.append(v2)
    return originals, views1, views2


def main() -> None:
    ensure_dir("results")
    set_seed()
    print(f"[task2] generating {NUM_EXAMPLES} augmentation examples")
    originals, views1, views2 = collect_examples()
    plot_augmentation_examples(
        originals,
        views1,
        views2,
        save_path=OUT_PATH,
        n_rows=NUM_EXAMPLES,
    )
    print(f"[task2] saved {OUT_PATH}")


if __name__ == "__main__":
    main()
