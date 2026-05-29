from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.augmentations import (
    build_eval_transform,
    build_supervised_train_transform,
)
from utils.dataset_splits import (
    SEED,
    IndexedSubset,
    build_labeled_train_dataset,
    build_loader,
    build_test_dataset,
    build_val_dataset,
    get_cifar10_train,
    load_indices,
)
from utils.metrics import (
    collect_predictions,
    evaluate_classifier,
    extract_features,
)
from utils.models import (
    CIFARResNet18,
    ClassifierModel,
    load_encoder_weights,
)
from utils.seed import set_seed
from utils.visualization import (
    ensure_dir,
    plot_accuracy_curves,
    save_2d_feature_plot,
)


BATCH_SIZE: int = 64
EPOCHS: int = 20
LEARNING_RATE: float = 3e-4

SIMCLR_ENCODER_CKPT: str = "models/simclr_encoder.pt"
FINETUNED_CKPT: str = "models/finetuned_model.pt"
FT_PLOT: str = "graphs/finetuning_accuracy.png"
RESULTS_JSON: str = "results/metrics_finetune.json"
TEST_PREDS_CSV: str = "results/test_predictions.csv"

PCA_TSNE_METHOD: str = "tsne"        # spec allows pca or tsne
PCA_TSNE_NUM_IMAGES: int = 1000
RAND_PCA_PNG: str = "results/random_encoder_pca_or_tsne.png"
SIMCLR_PCA_PNG: str = "results/simclr_encoder_pca_or_tsne.png"
FT_PCA_PNG: str = "results/finetuned_encoder_pca_or_tsne.png"


def _build_loaders():
    train_ds = build_labeled_train_dataset(transform=build_supervised_train_transform())
    val_ds = build_val_dataset(transform=build_eval_transform())
    test_ds = build_test_dataset(transform=build_eval_transform())
    return (
        build_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        build_loader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        build_loader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


def fine_tune_simclr(device: torch.device) -> Dict:
    
    model = ClassifierModel().to(device)
    load_encoder_weights(model.encoder, SIMCLR_ENCODER_CKPT, map_location=device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = _build_loaders()

    val_accuracies: List[float] = []
    best_state: Dict[str, torch.Tensor] | None = None
    best_val = -1.0
    best_epoch = 0

    print(f"[finetune] {EPOCHS} epochs on {device}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        seen = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * targets.size(0)
            seen += targets.size(0)
        train_loss = running_loss / max(seen, 1)

        val_metrics = evaluate_classifier(model, val_loader, device, criterion)
        val_accuracies.append(val_metrics["accuracy"])
        if val_metrics["accuracy"] > best_val:
            best_val = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"[finetune] {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']*100:.2f}% "
            f"time={time.time()-epoch_start:.1f}s"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[finetune] restored best val={best_val*100:.2f}% from epoch {best_epoch}")

    test_metrics = evaluate_classifier(model, test_loader, device, criterion)
    y_true, y_pred, probs = collect_predictions(model, test_loader, device)
    test_indices = load_indices("test")  # CIFAR-10 test indices in dataloader order

    return {
        "val_accuracies": val_accuracies,
        "best_val_accuracy": float(best_val),
        "best_epoch": int(best_epoch),
        "test_accuracy": float(test_metrics["accuracy"]),
        "model": model,
        "predictions": {
            "indices": test_indices,
            "true": y_true,
            "pred": y_pred,
            "probs": probs,
        },
    }


def get_fixed_val_subset(num_images: int = PCA_TSNE_NUM_IMAGES) -> IndexedSubset:
    
    val_indices = load_indices("val")
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(np.asarray(val_indices), size=num_images, replace=False).tolist()
    return IndexedSubset(get_cifar10_train(), chosen, transform=build_eval_transform())


def _visualize(encoder: nn.Module, loader, save_path: str, title: str, device: torch.device) -> None:
    features, labels = extract_features(encoder, loader, device)
    save_2d_feature_plot(
        features=features,
        labels=labels,
        out_path=save_path,
        method=PCA_TSNE_METHOD,
        title=title,
        seed=SEED,
    )


def generate_all_pca_tsne_plots(finetuned_model: ClassifierModel, device: torch.device) -> None:
    """Generate the 3 plots required by Task 8."""
    subset = get_fixed_val_subset()
    loader = build_loader(subset, batch_size=128, shuffle=False)

    rand_encoder = CIFARResNet18().to(device)
    _visualize(rand_encoder, loader, RAND_PCA_PNG,
               f"Random encoder ({PCA_TSNE_METHOD.upper()})", device)

    simclr_encoder = CIFARResNet18().to(device)
    load_encoder_weights(simclr_encoder, SIMCLR_ENCODER_CKPT, map_location=device)
    _visualize(simclr_encoder, loader, SIMCLR_PCA_PNG,
               f"SimCLR encoder ({PCA_TSNE_METHOD.upper()})", device)

    _visualize(finetuned_model.encoder, loader, FT_PCA_PNG,
               f"Fine-tuned encoder ({PCA_TSNE_METHOD.upper()})", device)


def write_test_predictions(predictions: Dict, out_path: str = TEST_PREDS_CSV) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    indices = predictions["indices"]
    y_true = predictions["true"]
    y_pred = predictions["pred"]
    probs = predictions["probs"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = (
            ["image_index", "true_label", "predicted_label"]
            + [f"prob_class_{c}" for c in range(10)]
        )
        writer.writerow(header)
        for img_idx, t, p, prob_row in zip(indices, y_true, y_pred, probs):
            writer.writerow([int(img_idx), int(t), int(p)] + [float(x) for x in prob_row])


def main() -> None:
    ensure_dir("graphs")
    ensure_dir("models")
    ensure_dir("results")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ft = fine_tune_simclr(device)

    torch.save(ft["model"].state_dict(), FINETUNED_CKPT)
    plot_accuracy_curves(
        {"SimCLR fine-tuning": ft["val_accuracies"]},
        FT_PLOT,
        title="Fine-tuning — validation accuracy",
    )

    generate_all_pca_tsne_plots(ft["model"], device)
    write_test_predictions(ft["predictions"])

    summary = {
        "simclr_finetune_test_acc": ft["test_accuracy"],
        "best_val_accuracy": ft["best_val_accuracy"],
        "best_epoch": ft["best_epoch"],
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Fine-tune + Task 8 results ===")
    print(f"Test accuracy        : {ft['test_accuracy']*100:.2f}%")
    print(f"Best val (epoch {ft['best_epoch']:>2}) : {ft['best_val_accuracy']*100:.2f}%")
    print(f"Saved model          -> {FINETUNED_CKPT}")
    print(f"Saved curve          -> {FT_PLOT}")
    print(f"Saved PCA/t-SNE      -> {RAND_PCA_PNG}, {SIMCLR_PCA_PNG}, {FT_PCA_PNG}")
    print(f"Saved predictions    -> {TEST_PREDS_CSV}")
    print(f"Saved summary        -> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
