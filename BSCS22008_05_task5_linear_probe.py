from __future__ import annotations
import json
import time
from typing import Dict, List
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.augmentations import (
    build_eval_transform,
    build_supervised_train_transform,
)
from utils.dataset_splits import (
    SEED,
    build_labeled_train_dataset,
    build_loader,
    build_test_dataset,
    build_val_dataset,
)
from utils.metrics import evaluate_classifier
from utils.models import (
    CIFARResNet18,
    ClassifierModel,
    load_encoder_weights,
)
from utils.seed import set_seed
from utils.visualization import ensure_dir, plot_accuracy_curves


BATCH_SIZE: int = 64
EPOCHS: int = 20
LEARNING_RATE: float = 3e-4

SIMCLR_ENCODER_CKPT: str = "models/simclr_encoder.pt"
PROBE_CKPT: str = "models/linear_probe.pt"
PROBE_PLOT: str = "graphs/linear_probe_accuracy.png"
RESULTS_JSON: str = "results/metrics_linear_probe.json"


def _build_loaders():
    train_ds = build_labeled_train_dataset(transform=build_supervised_train_transform())
    val_ds = build_val_dataset(transform=build_eval_transform())
    test_ds = build_test_dataset(transform=build_eval_transform())
    return (
        build_loader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        build_loader(val_ds, batch_size=BATCH_SIZE, shuffle=False),
        build_loader(test_ds, batch_size=BATCH_SIZE, shuffle=False),
    )


def train_linear_probe(encoder_init: str, device: torch.device) -> Dict:
    
    if encoder_init not in {"random", "simclr"}:
        raise ValueError(f"encoder_init must be 'random' or 'simclr', got {encoder_init!r}")

    model = ClassifierModel().to(device)
    if encoder_init == "simclr":
        load_encoder_weights(model.encoder, SIMCLR_ENCODER_CKPT, map_location=device)
    model.freeze_encoder()  # encoder.requires_grad=False, encoder.eval()

    optimizer = Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = _build_loaders()

    val_accuracies: List[float] = []
    print(f"[probe-{encoder_init}] training {EPOCHS} epochs on {device}")
    for epoch in range(1, EPOCHS + 1):
        model.classifier.train()
        # encoder stays in eval() mode set by freeze_encoder()
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
        print(
            f"[probe-{encoder_init}] {epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']*100:.2f}% "
            f"time={time.time()-epoch_start:.1f}s"
        )

    test_metrics = evaluate_classifier(model, test_loader, device, criterion)
    return {
        "val_accuracies": val_accuracies,
        "test_accuracy": float(test_metrics["accuracy"]),
        "model": model,
    }


def run_both_probes() -> Dict[str, Dict]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)
    results: Dict[str, Dict] = {}
    results["random"] = train_linear_probe("random", device)

    # Reseed before the second run so both probes see the same shuffle stream.
    set_seed(SEED)
    results["simclr"] = train_linear_probe("simclr", device)

    torch.save(results["simclr"]["model"].state_dict(), PROBE_CKPT)
    plot_accuracy_curves(
        {
            "Random encoder": results["random"]["val_accuracies"],
            "SimCLR encoder": results["simclr"]["val_accuracies"],
        },
        PROBE_PLOT,
        title="Linear probe — validation accuracy",
    )
    return results


def main() -> None:
    ensure_dir("graphs")
    ensure_dir("models")
    ensure_dir("results")

    results = run_both_probes()

    summary = {
        "random_linear_probe_test_acc": results["random"]["test_accuracy"],
        "simclr_linear_probe_test_acc": results["simclr"]["test_accuracy"],
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
    }
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Linear probe results ===")
    print(f"Random encoder test acc : {summary['random_linear_probe_test_acc']*100:.2f}%")
    print(f"SimCLR encoder test acc : {summary['simclr_linear_probe_test_acc']*100:.2f}%")
    print(f"Saved curves -> {PROBE_PLOT}")
    print(f"Saved probe  -> {PROBE_CKPT}")
    print(f"Saved metrics-> {RESULTS_JSON}")


if __name__ == "__main__":
    main()
