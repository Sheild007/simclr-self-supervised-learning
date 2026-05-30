from __future__ import annotations

import csv, json, time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.augmentations import build_eval_transform, build_supervised_train_transform
from utils.dataset_splits import SEED, IndexedSubset, build_labeled_train_dataset, build_loader, build_test_dataset, build_val_dataset, get_cifar10_train, load_indices
from utils.metrics import collect_predictions, evaluate_classifier, extract_features
from utils.models import CIFARResNet18, ClassifierModel, load_encoder_weights
from utils.seed import set_seed
from utils.visualization import ensure_dir, plot_accuracy_curves, save_2d_feature_plot


BATCH_SIZE = 64
EPOCHS = 20
LR = 3e-4

SIMCLR_CKPT = "models/simclr_encoder.pt"
FT_CKPT = "models/finetuned_model.pt"
FT_PLOT = "graphs/finetuning_accuracy.png"
RESULTS_OUT = "results/metrics_finetune.json"
PREDS_OUT = "results/test_predictions.csv"

VIZ_METHOD = "tsne"
VIZ_NUM = 1000
PLOT_RAND = "results/random_encoder_tsne.png"
PLOT_SIMCLR = "results/simclr_encoder_tsne.png"
PLOT_FT = "results/finetuned_encoder_tsne.png"


def make_loaders():
    tr = build_labeled_train_dataset(transform=build_supervised_train_transform())
    va = build_val_dataset(transform=build_eval_transform())
    te = build_test_dataset(transform=build_eval_transform())
    return (
        build_loader(tr, batch_size=BATCH_SIZE, shuffle=True),
        build_loader(va, batch_size=BATCH_SIZE, shuffle=False),
        build_loader(te, batch_size=BATCH_SIZE, shuffle=False),
    )


def fine_tune_simclr(device):
    model = ClassifierModel().to(device)
    load_encoder_weights(model.encoder, SIMCLR_CKPT, map_location=device)

    opt = Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    train_loader, val_loader, test_loader = make_loaders()

    accs: List[float] = []
    best_state, best_acc, best_epoch = None, -1.0, 0

    print(f"fine-tuning {EPOCHS} epochs on {device}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
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
        if vm["accuracy"] > best_acc:
            best_acc, best_epoch = vm["accuracy"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"  epoch {epoch:02d}/{EPOCHS}  train {run/max(seen,1):.4f}  "
              f"val {vm['loss']:.4f}  acc {vm['accuracy']*100:.2f}%  ({time.time()-t0:.1f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"reloaded best epoch {best_epoch} ({best_acc*100:.2f}%)")

    tm = evaluate_classifier(model, test_loader, device, crit)
    yt, yp, probs = collect_predictions(model, test_loader, device)
    test_indices = load_indices("test")

    return {
        "val_accuracies": accs,
        "best_val_accuracy": float(best_acc),
        "best_epoch": int(best_epoch),
        "test_accuracy": float(tm["accuracy"]),
        "model": model,
        "predictions": {"indices": test_indices, "true": yt, "pred": yp, "probs": probs},
    }


def get_fixed_val_subset(num=VIZ_NUM):
    val_idx = load_indices("val")
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(np.asarray(val_idx), size=num, replace=False).tolist()
    return IndexedSubset(get_cifar10_train(), chosen, transform=build_eval_transform())


def visualize(encoder, loader, out_path, title, device):
    feats, labels = extract_features(encoder, loader, device)
    save_2d_feature_plot(features=feats, labels=labels, out_path=out_path, title=title, seed=SEED)


def generate_all_tsne_plots(finetuned_model, device):
    subset = get_fixed_val_subset()
    loader = build_loader(subset, batch_size=128, shuffle=False)

    rand_enc = CIFARResNet18().to(device)
    visualize(rand_enc, loader, PLOT_RAND, f"Random encoder ({VIZ_METHOD.upper()})", device)

    sim_enc = CIFARResNet18().to(device)
    load_encoder_weights(sim_enc, SIMCLR_CKPT, map_location=device)
    visualize(sim_enc, loader, PLOT_SIMCLR, f"SimCLR encoder ({VIZ_METHOD.upper()})", device)

    visualize(finetuned_model.encoder, loader, PLOT_FT, f"Fine-tuned encoder ({VIZ_METHOD.upper()})", device)


def write_test_predictions(predictions, out_path=PREDS_OUT):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    indices = predictions["indices"]
    yt, yp, probs = predictions["true"], predictions["pred"], predictions["probs"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_index", "true_label", "predicted_label"] + [f"prob_class_{c}" for c in range(10)])
        for i, (t, p, row) in enumerate(zip(indices, zip(yt, yp), probs)):
            tt, pp = p
            w.writerow([int(i), int(tt), int(pp)] + [float(x) for x in row])


def main():
    for d in ("graphs", "models", "results"):
        ensure_dir(d)

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ft = fine_tune_simclr(device)
    torch.save(ft["model"].state_dict(), FT_CKPT)
    plot_accuracy_curves({"SimCLR fine-tuning": ft["val_accuracies"]}, FT_PLOT,
                         title="Fine-tuning (val accuracy)")

    generate_all_tsne_plots(ft["model"], device)
    write_test_predictions(ft["predictions"])

    summary = {
        "simclr_finetune_test_acc": ft["test_accuracy"],
        "best_val_accuracy": ft["best_val_accuracy"],
        "best_epoch": ft["best_epoch"],
        "epochs": EPOCHS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"test acc {ft['test_accuracy']*100:.2f}%  best val {ft['best_val_accuracy']*100:.2f}% (epoch {ft['best_epoch']})")
    print(f"saved {FT_CKPT}, {FT_PLOT}")
    print(f"saved {PLOT_RAND}, {PLOT_SIMCLR}, {PLOT_FT}")
    print(f"saved {PREDS_OUT}, {RESULTS_OUT}")


if __name__ == "__main__":
    main()
