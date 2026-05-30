from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import torch

from utils.seed import set_seed
from utils.dataset_splits import build_ssl_dataset, build_loader, get_cifar10_train, load_indices, IndexedSubset
from utils.augmentations import build_simclr_transform, TwoViewTransform
from utils.models import SimCLRModel, save_encoder
from utils.losses import DEFAULT_TEMPERATURE, cosine_similarity_matrix, build_positive_indices, nt_xent_loss, mean_same_image_similarity, mean_different_image_similarity
from utils.visualization import plot_loss_curves, plot_similarity_heatmap, ensure_dir


BATCH_SIZE = 64
EPOCHS = 50
LR = 3e-4
TEMPERATURE = DEFAULT_TEMPERATURE
NUM_WORKERS = 0

ENCODER_CKPT = "models/simclr_encoder.pt"
LOSS_PLOT = "graphs/simclr_pretraining_loss.png"
SIM_AFTER_PLOT = "results/similarity_matrix_after_training.png"
SIM_METRICS_OUT = "results/metrics_similarity_after.json"


def task4_sanity_check():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")

    model = SimCLRModel().to(device).eval()
    n_total = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_proj = sum(p.numel() for p in model.projection_head.parameters())
    print(f"params: total {n_total:,}  encoder {n_enc:,}  projection {n_proj:,}")

    x = torch.randn(8, 3, 32, 32, device=device)
    with torch.no_grad():
        z = model(x)
    print(f"forward shape {tuple(z.shape)} (expected (8, 128))")
    assert z.shape == (8, 128)

    pos = build_positive_indices(4).tolist()
    print(f"positive indices N=4: {pos}")
    assert pos == [4, 5, 6, 7, 0, 1, 2, 3]

    print("nt-xent on random projections:")
    for n in (4, 16, 64):
        z_rand = torch.randn(2 * n, 128, device=device)
        loss = nt_xent_loss(z_rand, temperature=TEMPERATURE).item()
        print(f"  N={n:3d}  loss={loss:.4f}  log(2N-1)={math.log(2*n-1):.4f}")
    print("sanity check passed")


def simclr_train_one_epoch(model, loader, opt, device, tau):
    model.train()
    total, n = 0.0, 0
    for (v1, v2), _ in loader:
        x = torch.cat([v1, v2], dim=0).to(device, non_blocking=True)
        z = model(x)
        loss = nt_xent_loss(z, temperature=tau)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        bs = v1.size(0)
        total += loss.item() * bs
        n += bs
    return total / max(n, 1)


def compute_post_training_similarity(model, device, batch_size=64):
    two_view = TwoViewTransform(build_simclr_transform())
    indices = load_indices("val")[:batch_size]
    dataset = IndexedSubset(get_cifar10_train(), indices, transform=two_view)
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False)

    encoder = model.encoder.eval()
    (v1, v2), _ = next(iter(loader))
    x = torch.cat([v1, v2], dim=0).to(device)
    with torch.no_grad():
        h = encoder(x)

    same = mean_same_image_similarity(h, batch_size)
    diff = mean_different_image_similarity(h, batch_size)
    plot_similarity_heatmap(cosine_similarity_matrix(h), out_path=SIM_AFTER_PLOT,
                            title=f"Cosine similarity (SimCLR encoder, N={batch_size})",
                            batch_size=batch_size)
    return {
        "same_view_similarity_after": float(same),
        "different_image_similarity_after": float(diff),
        "gap_after": float(same - diff),
    }


def pretrain_simclr(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, tau=TEMPERATURE, device=None):
    set_seed()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}  epochs {epochs}  batch {batch_size}  lr {lr}  tau {tau}")

    two_view = TwoViewTransform(build_simclr_transform())
    dataset = build_ssl_dataset(two_view)
    loader = build_loader(dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    print(f"dataset {len(dataset)}  batches/epoch {len(loader)}")

    model = SimCLRModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    wall = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = simclr_train_one_epoch(model, loader, opt, device, tau)
        losses.append(loss)
        print(f"epoch {epoch:02d}/{epochs}  loss {loss:.4f}  ({time.time()-t0:.1f}s)")

    save_encoder(model.encoder, ENCODER_CKPT)
    print(f"saved {ENCODER_CKPT}")

    ensure_dir(Path(LOSS_PLOT).parent)
    plot_loss_curves(losses, losses, out_path=LOSS_PLOT, title=f"SimCLR pretraining ({epochs} epochs)")
    print(f"saved {LOSS_PLOT}")

    sims = compute_post_training_similarity(model, device, batch_size=batch_size)
    print(f"same view after  {sims['same_view_similarity_after']:.4f}")
    print(f"diff image after {sims['different_image_similarity_after']:.4f}")
    print(f"gap after        {sims['gap_after']:+.4f}")

    out = {
        "epochs_run": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "temperature": tau,
        "total_minutes": round((time.time() - wall) / 60.0, 2),
        "final_loss": float(losses[-1]),
        **sims,
    }
    Path(SIM_METRICS_OUT).write_text(json.dumps(out, indent=2))
    print(f"saved {SIM_METRICS_OUT}")
    return {"epoch_losses": losses, **out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run 50-epoch simclr pretraining")
    args = parser.parse_args()

    for d in ("graphs", "results", "models"):
        ensure_dir(d)

    if args.train:
        pretrain_simclr()
    else:
        task4_sanity_check()


if __name__ == "__main__":
    main()
