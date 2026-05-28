"""Basic metric helpers for Assignment 5."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def top1_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def compute_accuracy(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_confusion_matrix(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(10)))


def per_class_accuracy(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> dict[str, float]:
    cm = compute_confusion_matrix(y_true, y_pred)
    result: dict[str, float] = {}
    for i, cls_name in enumerate(CIFAR10_CLASSES):
        denom = cm[i].sum()
        result[cls_name] = float(cm[i, i] / denom) if denom else 0.0
    return result


def save_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    out_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cm = compute_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CIFAR10_CLASSES)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


@torch.no_grad()
def evaluate_classifier(model: nn.Module,loader: Iterable,device: torch.device,criterion: nn.Module | None = None,) -> dict[str, float]:
    
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += batch_size

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
    }


@torch.no_grad()
def collect_predictions(model: nn.Module,loader: Iterable,device: torch.device,) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    model.eval()

    y_true_list: List[int] = []
    y_pred_list: List[int] = []
    probs_list: List[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)

        y_true_list.extend(targets.tolist())
        y_pred_list.extend(preds.tolist())
        probs_list.append(probs)

    y_true = np.asarray(y_true_list, dtype=np.int64)
    y_pred = np.asarray(y_pred_list, dtype=np.int64)
    if probs_list:
        probabilities = np.concatenate(probs_list, axis=0)
    else:
        probabilities = np.zeros((0, 10), dtype=np.float32)
    return y_true, y_pred, probabilities
