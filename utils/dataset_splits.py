"""Fixed CIFAR-10 split loading for Assignment 5.

This module is the single source of truth for turning the four TA-provided
split files under ``splits/`` into PyTorch datasets and DataLoaders. Every
task script imports its data-building helpers from here so that the same
fixed splits are used everywhere.

Key rules (per ASSIGNMENT.md §3):
- The four split files (``train_labeled_10percent``, ``train_ssl_unlabeled``,
  ``val``, ``test``) are the only allowed way to split the data.
- ``random_split()`` and homemade splits are forbidden.
- Indices in the train/val files refer to the official CIFAR-10 *train*
  array; indices in ``test.txt`` refer to the official CIFAR-10 *test* array.
- Seed ``2026`` is used for any randomness introduced by this module
  (e.g. DataLoader shuffling).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


# ---------------------------------------------------------------------------
# Project-wide defaults
# ---------------------------------------------------------------------------

SEED: int = 2026

DATA_ROOT: Path = Path("data")
SPLITS_DIR: Path = Path("splits")

_SPLIT_FILES: dict[str, str] = {
    "train_labeled_10percent": "train_labeled_10percent.txt",
    "train_ssl_unlabeled": "train_ssl_unlabeled.txt",
    "val": "val.txt",
    "test": "test.txt",
}

# Indices in these splits index into the official CIFAR-10 TRAIN array.
_TRAIN_SPLITS: frozenset[str] = frozenset(
    {"train_labeled_10percent", "train_ssl_unlabeled", "val"}
)
# Indices in this split index into the official CIFAR-10 TEST array.
_TEST_SPLITS: frozenset[str] = frozenset({"test"})


# ---------------------------------------------------------------------------
# Index file loading
# ---------------------------------------------------------------------------

def _resolve_split_path(name_or_path: str | Path) -> Path:
    """Resolve a short split name OR a direct path to a split file.

    Short names (e.g. ``"val"``) are looked up under ``SPLITS_DIR``; any
    string ending in ``.txt`` or pointing to an existing file is returned
    as-is so tests can point at custom locations.
    """
    candidate = Path(name_or_path)
    if candidate.suffix == ".txt" or candidate.exists():
        return candidate
    key = str(name_or_path)
    if key in _SPLIT_FILES:
        return SPLITS_DIR / _SPLIT_FILES[key]
    raise KeyError(
        f"Unknown split '{name_or_path}'. Expected one of "
        f"{sorted(_SPLIT_FILES)} or a path to a .txt file."
    )


def load_indices(name_or_path: str | Path) -> list[int]:
    """Read integer indices, one per line, from a split file.

    Args:
        name_or_path: Either a short name like ``"val"`` or a direct path
            to a ``.txt`` file containing one integer index per line.

    Returns:
        Indices in their on-disk order. Order is intentionally preserved
        because some callers rely on it (e.g. Task 2 picks the first N
        indices for the augmentation visualization).
    """
    path = _resolve_split_path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file '{path}' not found. Place the four TA-provided "
            f"split files under '{SPLITS_DIR}/'."
        )
    text = path.read_text(encoding="utf-8")
    return [int(line) for line in text.splitlines() if line.strip()]


read_split_indices = load_indices  # legacy alias


# ---------------------------------------------------------------------------
# Underlying CIFAR-10 datasets (transform=None → returns PIL images)
# ---------------------------------------------------------------------------

def get_cifar10_train(
    data_root: str | Path = DATA_ROOT,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
) -> CIFAR10:
    """Return the official CIFAR-10 *train* set.

    By default no transform is applied so callers get raw PIL images and
    can wrap them in ``IndexedSubset(..., transform=...)`` themselves. This
    matches how Tasks 2/3/4 expect to consume the data.
    """
    return CIFAR10(
        root=str(data_root),
        train=True,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


def get_cifar10_test(
    data_root: str | Path = DATA_ROOT,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
) -> CIFAR10:
    """Return the official CIFAR-10 *test* set."""
    return CIFAR10(
        root=str(data_root),
        train=False,
        transform=transform,
        target_transform=target_transform,
        download=download,
    )


# ---------------------------------------------------------------------------
# IndexedSubset
# ---------------------------------------------------------------------------

class IndexedSubset(Dataset):
    """A subset of a CIFAR-10 dataset with its own transform applied lazily.

    The wrapped ``base_dataset`` is expected to be a torchvision CIFAR-10
    instance with ``transform=None`` so it returns PIL images. The transform
    passed here is applied at ``__getitem__`` time, which lets different
    tasks reuse the same underlying CIFAR-10 with different augmentation
    pipelines (CIFAR-10 fits in RAM, so the duplicate base instance is cheap).

    The transform may be a ``TwoViewTransform`` returning ``(view1, view2)``;
    in that case ``__getitem__`` yields ``((view1, view2), target)`` and the
    default DataLoader collate handles the shape correctly.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: Sequence[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices: list[int] = list(indices)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        real_idx = self.indices[idx]
        image, target = self.base_dataset[real_idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


# ---------------------------------------------------------------------------
# High-level dataset builders (used by Tasks 1, 5, 6, 7)
# ---------------------------------------------------------------------------

def _build_indexed_dataset(
    split_name: str,
    transform: Optional[Callable],
    data_root: str | Path,
) -> IndexedSubset:
    indices = load_indices(split_name)
    if split_name in _TRAIN_SPLITS:
        base = get_cifar10_train(data_root=data_root)
    elif split_name in _TEST_SPLITS:
        base = get_cifar10_test(data_root=data_root)
    else:
        raise KeyError(f"Unknown split: {split_name}")
    return IndexedSubset(base, indices, transform=transform)


def build_labeled_train_dataset(
    transform: Optional[Callable] = None,
    data_root: str | Path = DATA_ROOT,
) -> IndexedSubset:
    """10% labeled split — Task 1, linear probe, fine-tune."""
    return _build_indexed_dataset("train_labeled_10percent", transform, data_root)


def build_val_dataset(
    transform: Optional[Callable] = None,
    data_root: str | Path = DATA_ROOT,
) -> IndexedSubset:
    """Validation split — model selection only."""
    return _build_indexed_dataset("val", transform, data_root)


def build_test_dataset(
    transform: Optional[Callable] = None,
    data_root: str | Path = DATA_ROOT,
) -> IndexedSubset:
    """Test split — final reporting only."""
    return _build_indexed_dataset("test", transform, data_root)


def build_ssl_dataset(
    two_view_transform: Callable,
    data_root: str | Path = DATA_ROOT,
) -> IndexedSubset:
    """Unlabeled SSL split — SimCLR pretraining (Task 4 / Task 5).

    The CIFAR-10 dataset still returns labels, but the SimCLR training loop
    must not use them. Pass a ``TwoViewTransform`` so each item yields
    ``((view1, view2), label)``.
    """
    return _build_indexed_dataset(
        "train_ssl_unlabeled", two_view_transform, data_root
    )


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_loader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: Optional[bool] = None,
    seed: int = SEED,
) -> DataLoader:
    """Build a reproducible DataLoader.

    A ``torch.Generator`` seeded with ``seed`` is wired to the DataLoader so
    the shuffle order is deterministic across runs when the global seed is
    the same.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Legacy helpers kept for backwards compatibility
# ---------------------------------------------------------------------------

def get_cifar10_subset(
    data_root: str | Path,
    split_file: str | Path,
    train: bool,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
) -> Dataset:
    """Legacy: return a CIFAR-10 subset for a given split file path.

    Prefer ``build_labeled_train_dataset`` / ``build_val_dataset`` /
    ``build_test_dataset`` / ``build_ssl_dataset``.
    """
    base_factory = get_cifar10_train if train else get_cifar10_test
    base = base_factory(data_root=data_root, download=download)
    indices = load_indices(split_file)
    return IndexedSubset(
        base,
        indices,
        transform=transform,
        target_transform=target_transform,
    )


class TwoViewDataset(Dataset):
    """Legacy two-view wrapper. Prefer ``IndexedSubset`` + ``TwoViewTransform``."""

    def __init__(self, base_dataset: Dataset, two_view_transform: Callable) -> None:
        self.base_dataset = base_dataset
        self.two_view_transform = two_view_transform

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, target = self.base_dataset[idx]
        view1, view2 = self.two_view_transform(image)
        return view1, view2, target


__all__ = [
    "SEED",
    "DATA_ROOT",
    "SPLITS_DIR",
    "load_indices",
    "read_split_indices",
    "get_cifar10_train",
    "get_cifar10_test",
    "IndexedSubset",
    "build_labeled_train_dataset",
    "build_val_dataset",
    "build_test_dataset",
    "build_ssl_dataset",
    "build_loader",
    "get_cifar10_subset",
    "TwoViewDataset",
]
