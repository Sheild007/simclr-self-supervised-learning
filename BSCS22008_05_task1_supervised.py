from __future__ import annotations

import json, time
from pathlib import Path

import torch
import torch.nn as nn

from utils.seed import set_seed
from utils.dataset_splits import SEED, build_labeled_train_dataset, build_val_dataset, build_test_dataset, build_loader
from utils.augmentations import build_supervised_train_transform, build_eval_transform
from utils.models import ClassifierModel
from utils.metrics import evaluate_classifier, collect_predictions, compute_confusion_matrix
from utils.visualization import plot_loss_curves, plot_confusion_matrix, ensure_dir


BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4
PATIENCE = 5
MIN_DELTA = 1e-4
NUM_WORKERS = 0

LOSS_PLOT = "graphs/supervised_loss.png"
CM_PLOT = "results/supervised_confusion_matrix.png"
METRICS_OUT = "results/metrics_supervised.json"
BEST_CKPT = "models/supervised_best.pt"


def train_one_epoch(model, loader, opt, crit, device):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = crit(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * y.size(0)
        n += y.size(0)
    return total / max(n, 1)


def run_supervised_baseline(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, patience=PATIENCE, min_delta=MIN_DELTA, device=None):
    set_seed()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = build_labeled_train_dataset(transform=build_supervised_train_transform())
    val_ds = build_val_dataset(transform=build_eval_transform())
    test_ds = build_test_dataset(transform=build_eval_transform())
    print(f"train {len(train_ds)}  val {len(val_ds)}  test {len(test_ds)}  device {device}")

    train_loader = build_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = build_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = build_loader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    model = ClassifierModel(num_classes=10).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs = [], [], []
    best_acc, best_epoch, no_improve = 0.0, 0, 0
    early_stopped = False
    last_epoch = 0

    ensure_dir(Path(BEST_CKPT).parent)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(model, train_loader, opt, crit, device)
        vm = evaluate_classifier(model, val_loader, device, crit)
        train_losses.append(tl)
        val_losses.append(vm["loss"])
        val_accs.append(vm["accuracy"])
        last_epoch = epoch

        improved = vm["accuracy"] > best_acc + min_delta
        tag = ""
        if improved:
            best_acc, best_epoch, no_improve = vm["accuracy"], epoch, 0
            torch.save(model.state_dict(), BEST_CKPT)
            tag = "  *"
        else:
            no_improve += 1

        print(f"epoch {epoch:02d}/{epochs}  train {tl:.4f}  val {vm['loss']:.4f}  acc {vm['accuracy']*100:.2f}%  ({time.time()-t0:.1f}s){tag}")

        if no_improve >= patience:
            print(f"no improvement for {patience} epochs, stopping at {epoch}")
            early_stopped = True
            break

    if Path(BEST_CKPT).exists():
        print(f"reloading best from epoch {best_epoch} ({best_acc*100:.2f}%)")
        model.load_state_dict(torch.load(BEST_CKPT, map_location=device))

    tm = evaluate_classifier(model, test_loader, device, crit)
    print(f"test loss {tm['loss']:.4f}  test acc {tm['accuracy']*100:.2f}%")

    ensure_dir(Path(LOSS_PLOT).parent)
    plot_loss_curves(train_losses, val_losses, out_path=LOSS_PLOT,
                     title=f"Supervised baseline (10% labels), {last_epoch}/{epochs} epochs",
                     val_accuracies=val_accs)

    yt, yp, _ = collect_predictions(model, test_loader, device)
    cm = compute_confusion_matrix(yt, yp)
    ensure_dir(Path(CM_PLOT).parent)
    plot_confusion_matrix(cm, out_path=CM_PLOT,
                          title=f"Supervised ResNet-18 test acc {tm['accuracy']*100:.2f}%")
    print(f"saved {LOSS_PLOT}, {CM_PLOT}")

    out = {
        "test_accuracy": float(tm["accuracy"]),
        "test_loss": float(tm["loss"]),
        "best_val_accuracy": float(best_acc),
        "best_val_epoch": int(best_epoch),
        "epochs_run": int(last_epoch),
        "epochs_planned": int(epochs),
        "early_stopped": bool(early_stopped),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "batch_size": int(batch_size),
        "learning_rate": float(lr),
        "seed": int(SEED),
        "best_checkpoint": BEST_CKPT,
    }
    ensure_dir(Path(METRICS_OUT).parent)
    Path(METRICS_OUT).write_text(json.dumps(out, indent=2))
    print(f"saved {METRICS_OUT}")
    return out


def main():
    for d in ("graphs", "results", "models"):
        ensure_dir(d)
    out = run_supervised_baseline()
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
