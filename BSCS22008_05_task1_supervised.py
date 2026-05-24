from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.seed import set_seed
from utils.dataset_splits import SEED, build_labeled_train_dataset, build_val_dataset, build_test_dataset, build_loader
from utils.augmentations import build_supervised_train_transform, build_eval_transform
from utils.models import ClassifierModel
from utils.metrics import evaluate_classifier, collect_predictions, compute_confusion_matrix
from utils.visualization import plot_loss_curves, plot_confusion_matrix, ensure_dir


BATCH_SIZE: int = 64
EPOCHS: int = 20
LEARNING_RATE: float = 3e-4
NUM_WORKERS: int = 0

# Early-stopping settings (val accuracy is the monitored metric)
PATIENCE: int = 5          # stop if val_acc does not improve for this many epochs
MIN_DELTA: float = 1e-4    # must improve by at least this much to count as progress

LOSS_PLOT_PATH: str = "graphs/supervised_loss.png"
CONFMAT_PATH: str = "results/supervised_confusion_matrix.png"
METRICS_PARTIAL_PATH: str = "results/metrics_supervised.json"
BEST_CKPT_PATH: str = "models/supervised_best.pt"

CIFAR10_CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def run_supervised_baseline(
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
    device: torch.device | None = None,
) -> Dict[str, float]:
    set_seed()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[task1] device={device}, epochs={epochs}, batch_size={batch_size}, "
        f"patience={patience}, min_delta={min_delta}"
    )

    train_transform = build_supervised_train_transform()
    eval_transform = build_eval_transform()

    train_ds = build_labeled_train_dataset(transform=train_transform)
    val_ds = build_val_dataset(transform=eval_transform)
    test_ds = build_test_dataset(transform=eval_transform)
    print(f"[task1] dataset sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = build_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=False)
    val_loader = build_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = build_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = ClassifierModel(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses: List[float] = []
    val_losses: List[float] = []
    val_accuracies: List[float] = []
    best_val_accuracy = 0.0
    best_val_epoch = 0
    epochs_without_improvement = 0
    early_stopped = False
    epochs_run = 0

    ensure_dir(Path(BEST_CKPT_PATH).parent)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_classifier(model, val_loader, device, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_metrics["loss"])
        val_accuracies.append(val_metrics["accuracy"])
        epochs_run = epoch

        improved = val_metrics["accuracy"] > best_val_accuracy + min_delta
        marker = ""
        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_CKPT_PATH)
            marker = "  [best->saved]"
        else:
            epochs_without_improvement += 1

        print(
            f"[task1] epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']*100:.2f}% | "
            f"no_improve={epochs_without_improvement}/{patience} | "
            f"time={time.time() - epoch_start:.1f}s"
            f"{marker}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"[task1] early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            early_stopped = True
            break

    if Path(BEST_CKPT_PATH).exists():
        print(
            f"[task1] reloading best checkpoint from {BEST_CKPT_PATH} "
            f"(epoch {best_val_epoch}, val_acc={best_val_accuracy*100:.2f}%)"
        )
        model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=device))

    test_metrics = evaluate_classifier(model, test_loader, device, criterion)
    print(
        f"[task1] BEST CKPT TEST | test_loss={test_metrics['loss']:.4f} | "
        f"test_acc={test_metrics['accuracy']*100:.2f}%"
    )

    ensure_dir(Path(LOSS_PLOT_PATH).parent)
    plot_loss_curves(
        train_losses,
        val_losses,
        out_path=LOSS_PLOT_PATH,
        title=f"Supervised baseline (10% labels) — {epochs_run}/{epochs} epochs",
        val_accuracies=val_accuracies,
    )
    print(f"[task1] saved {LOSS_PLOT_PATH}")

    y_true, y_pred, _ = collect_predictions(model, test_loader, device)
    cm = compute_confusion_matrix(y_true, y_pred)
    ensure_dir(Path(CONFMAT_PATH).parent)
    plot_confusion_matrix(
        cm,
        out_path=CONFMAT_PATH,
        title=f"Supervised ResNet-18 on CIFAR-10 test (acc={test_metrics['accuracy']*100:.2f}%)",
    )
    print(f"[task1] saved {CONFMAT_PATH}")

    metrics: Dict[str, float] = {
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_loss": float(test_metrics["loss"]),
        "best_val_accuracy": float(best_val_accuracy),
        "best_val_epoch": int(best_val_epoch),
        "epochs_run": int(epochs_run),
        "epochs_planned": int(epochs),
        "early_stopped": bool(early_stopped),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "seed": int(SEED),
        "best_checkpoint": BEST_CKPT_PATH,
    }

    ensure_dir(Path(METRICS_PARTIAL_PATH).parent)
    Path(METRICS_PARTIAL_PATH).write_text(json.dumps(metrics, indent=2))
    print(f"[task1] saved {METRICS_PARTIAL_PATH}")

    return metrics


def main() -> None:
    ensure_dir("graphs")
    ensure_dir("results")
    ensure_dir("models")
    metrics = run_supervised_baseline()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
