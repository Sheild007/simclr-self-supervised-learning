"""Utilities for loading fixed split files.

This file intentionally does not implement SimCLR. It only helps students load
CIFAR-10 subsets from the instructor-provided split files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10


SEED = 2026
DATA_ROOT = Path("data")
SPLITS_DIR = Path("splits")

_SPLIT_FILES = {
    "train_labeled_10percent": "train_labeled_10percent.txt",
    "train_ssl_unlabeled": "train_ssl_unlabeled.txt",
    "val": "val.txt",
    "test": "test.txt",
}
_TRAIN_SPLITS = frozenset({"train_labeled_10percent", "train_ssl_unlabeled", "val"})
_TEST_SPLITS = frozenset({"test"})


def _resolve_split_path(name_or_path):
    candidate = Path(name_or_path)
    if candidate.suffix == ".txt" or candidate.exists():
        return candidate
    key = str(name_or_path)
    if key in _SPLIT_FILES:
        return SPLITS_DIR / _SPLIT_FILES[key]
    raise KeyError(f"Unknown split '{name_or_path}'. Expected one of {sorted(_SPLIT_FILES)} or a path to a .txt file.")


def load_indices(name_or_path):
    path = _resolve_split_path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Split file '{path}' not found. Place the four TA-provided split files under '{SPLITS_DIR}/'.")
    return [int(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_split_indices(path: str | Path) -> list[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [int(line) for line in lines if line]


def get_cifar10_train(data_root=DATA_ROOT, transform=None, target_transform=None, download=False):
    return CIFAR10(root=str(data_root), train=True, transform=transform, target_transform=target_transform, download=download)


def get_cifar10_test(data_root=DATA_ROOT, transform=None, target_transform=None, download=False):
    return CIFAR10(root=str(data_root), train=False, transform=transform, target_transform=target_transform, download=download)


class IndexedSubset(Dataset):
    def __init__(self, base_dataset, indices, transform=None, target_transform=None):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, target = self.base_dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def _build_indexed_dataset(split_name, transform, data_root):
    indices = load_indices(split_name)
    if split_name in _TRAIN_SPLITS:
        base = get_cifar10_train(data_root=data_root)
    elif split_name in _TEST_SPLITS:
        base = get_cifar10_test(data_root=data_root)
    else:
        raise KeyError(f"Unknown split: {split_name}")
    return IndexedSubset(base, indices, transform=transform)


def build_labeled_train_dataset(transform=None, data_root=DATA_ROOT):
    return _build_indexed_dataset("train_labeled_10percent", transform, data_root)


def build_val_dataset(transform=None, data_root=DATA_ROOT):
    return _build_indexed_dataset("val", transform, data_root)


def build_test_dataset(transform=None, data_root=DATA_ROOT):
    return _build_indexed_dataset("test", transform, data_root)


def build_ssl_dataset(two_view_transform, data_root=DATA_ROOT):
    return _build_indexed_dataset("train_ssl_unlabeled", two_view_transform, data_root)


def build_loader(dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False, pin_memory=None, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory, generator=generator)


def get_cifar10_subset(
    data_root: str | Path,
    split_file: str | Path,
    train: bool,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
) -> Dataset:
    """Return a CIFAR-10 subset according to a fixed split file.

    Args:
        data_root: CIFAR-10 data root.
        split_file: Path to a txt file containing integer indices.
        train: Use CIFAR-10 official train split if True, official test split if False.
        transform: Optional image transform.
        target_transform: Optional label transform.
        download: Download CIFAR-10 if not present.
    """
    dataset = CIFAR10(
        root=str(data_root),
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )
    indices = read_split_indices(split_file)
    return Subset(dataset, indices)


class TwoViewDataset(Dataset):
    """Wrap a dataset so it returns two augmented views and the original target.

    For SimCLR pretraining, students should ignore the target in the training loop.
    This wrapper is provided as a simple data utility, not as a SimCLR implementation.
    """

    def __init__(self, base_dataset: Dataset, two_view_transform: Callable):
        self.base_dataset = base_dataset
        self.two_view_transform = two_view_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[idx]
        view1, view2 = self.two_view_transform(image)
        return view1, view2, target


__all__ = [
    "SEED", "DATA_ROOT", "SPLITS_DIR",
    "load_indices", "read_split_indices",
    "get_cifar10_train", "get_cifar10_test", "IndexedSubset",
    "build_labeled_train_dataset", "build_val_dataset", "build_test_dataset",
    "build_ssl_dataset", "build_loader",
    "get_cifar10_subset", "TwoViewDataset",
]
