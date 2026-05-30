from __future__ import annotations

import json, time
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import Adam

from utils.augmentations import build_eval_transform, build_supervised_train_transform
from utils.dataset_splits import SEED, build_labeled_train_dataset, build_loader, build_test_dataset, build_val_dataset
from utils.metrics import evaluate_classifier
from utils.models import ClassifierModel, load_encoder_weights
from utils.seed import set_seed
from utils.visualization import ensure_dir, plot_accuracy_curves


BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4

SIMCLR_CKPT = "models/simclr_encoder.pt"
PROBE_CKPT = "models/linear_probe.pt"
PROBE_PLOT = "graphs/linear_probe_accuracy.png"
RESULTS_OUT = "results/metrics_linear_probe.json"


def make_loaders():
    tr = build_labeled_train_dataset(transform=build_supervised_train_transform())
    va = build_val_dataset(transform=build_eval_transform())
    te = build_test_dataset(transform=build_eval_transform())
    return (
        build_loader(tr, batch_size=BATCH_SIZE, shuffle=True),
        build_loader(va, batch_size=BATCH_SIZE, shuffle=False),
        build_loader(te, batch_size=BATCH_SIZE, shuffle=False),
    )


def train_linear_probe(encoder_init, device):
    if encoder_init not in {"random", "simclr"}:
        raise ValueError(f"encoder_init must be 'random' or 'simclr', got {encoder_init!r}")

    model = ClassifierModel().to(device)
    if encoder_init == "simclr":
        load_encoder_weights(model.encoder, SIMCLR_CKPT, map_location=device)
    model.freeze_encoder()

    opt = Adam(model.classifier.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = make_loaders()

    accs: List[float] = []
    print(f"probe {encoder_init}: {EPOCHS} epochs on {device}")
    for epoch in range(1, EPOCHS + 1):
        model.classifier.train()
        t0 = time.time()
        run, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item() * y.size(0)
            seen += y.size(0)
        vm = evaluate_classifier(model, val_loader, device, crit)
        accs.append(vm["accuracy"])
        print(f"  epoch {epoch:02d}/{EPOCHS}  train {run/max(seen,1):.4f}  "
              f"val {vm['loss']:.4f}  acc {vm['accuracy']*100:.2f}%  ({time.time()-t0:.1f}s)")

    tm = evaluate_classifier(model, test_loader, device, crit)
    return {"val_accuracies": accs, "test_accuracy": float(tm["accuracy"]), "model": model}


def run_both_probes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(SEED)
    out: Dict[str, Dict] = {}
    out["random"] = train_linear_probe("random", device)

    set_seed(SEED)
    out["simclr"] = train_linear_probe("simclr", device)

    torch.save(out["simclr"]["model"].state_dict(), PROBE_CKPT)
    plot_accuracy_curves(
        {"Random encoder": out["random"]["val_accuracies"],
         "SimCLR encoder": out["simclr"]["val_accuracies"]},
        PROBE_PLOT,
        title="Linear probe (val accuracy)",
    )
    return out


def main():
    for d in ("graphs", "models", "results"):
        ensure_dir(d)

    out = run_both_probes()

    summary = {
        "random_linear_probe_test_acc": out["random"]["test_accuracy"],
        "simclr_linear_probe_test_acc": out["simclr"]["test_accuracy"],
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"random test {summary['random_linear_probe_test_acc']*100:.2f}%")
    print(f"simclr test {summary['simclr_linear_probe_test_acc']*100:.2f}%")
    print(f"saved {PROBE_PLOT}, {PROBE_CKPT}, {RESULTS_OUT}")


if __name__ == "__main__":
    main()
