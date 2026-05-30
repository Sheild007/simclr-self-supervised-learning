from __future__ import annotations

import torchvision.transforms as T

from utils.seed import set_seed
from utils.dataset_splits import get_cifar10_train, load_indices
from utils.augmentations import build_simclr_transform, TwoViewTransform
from utils.visualization import plot_augmentation_examples, ensure_dir


NUM_EXAMPLES = 10
OUT_PATH = "results/augmentation_examples.png"


def collect_examples(num_examples=NUM_EXAMPLES):
    cifar = get_cifar10_train()
    indices = load_indices("train_ssl_unlabeled")[:num_examples]
    two_view = TwoViewTransform(build_simclr_transform())
    to_tensor = T.ToTensor()

    originals, v1s, v2s = [], [], []
    for idx in indices:
        img, _ = cifar[idx]
        a, b = two_view(img)
        originals.append(to_tensor(img))
        v1s.append(a)
        v2s.append(b)
    return originals, v1s, v2s


def main():
    ensure_dir("results")
    set_seed()
    print(f"generating {NUM_EXAMPLES} augmentation examples")
    originals, v1s, v2s = collect_examples()
    plot_augmentation_examples(originals, v1s, v2s, save_path=OUT_PATH, n_rows=NUM_EXAMPLES)
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main()
