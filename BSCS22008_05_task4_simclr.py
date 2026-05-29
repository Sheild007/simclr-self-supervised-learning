from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from utils.seed import set_seed
from utils.dataset_splits import (
    SEED,
    build_ssl_dataset,
    build_loader,
    get_cifar10_train,
    load_indices,
    IndexedSubset,
)
from utils.augmentations import build_simclr_transform, TwoViewTransform
from utils.models import (
    CIFARResNet18,
    ProjectionHead,
    SimCLRModel,
    save_encoder,
)
from utils.losses import (
    DEFAULT_TEMPERATURE,
    cosine_similarity_matrix,
    build_positive_indices,
    nt_xent_loss,
    mean_same_image_similarity,
    mean_different_image_similarity,
)
from utils.visualization import (
    plot_loss_curves,
    plot_similarity_heatmap,
    ensure_dir,
)



BATCH_SIZE: int = 64
EPOCHS: int = 50
LEARNING_RATE: float = 3e-4
TEMPERATURE: float = DEFAULT_TEMPERATURE
NUM_WORKERS: int = 0

ENCODER_CKPT: str = "models/simclr_encoder.pt"
LOSS_PLOT: str = "graphs/simclr_pretraining_loss.png"
SIM_MATRIX_AFTER_PLOT: str = "results/similarity_matrix_after_training.png"
SIM_METRICS_PATH: str = "results/metrics_similarity_after.json"


def task4_sanity_check() -> None:
   
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[task4] device={device}")

    print("\n[task4.1] Building SimCLRModel...")
    model = SimCLRModel().to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_proj = sum(p.numel() for p in model.projection_head.parameters())
    print(f"  total params       : {n_params:,}")
    print(f"  encoder params     : {n_enc:,}")
    print(f"  projection params  : {n_proj:,}")

    x = torch.randn(8, 3, 32, 32, device=device)
    with torch.no_grad():
        z = model(x)
    print(f"  SimCLRModel(x).shape = {tuple(z.shape)}  (expected (8, 128))")
    assert z.shape == (8, 128)

    print("\n[task4.2] Positive-pair indices for N=4:")
    pos_n4 = build_positive_indices(4).tolist()
    print(f"  i   : {list(range(8))}")
    print(f"  pos : {pos_n4}")
    print("  -> matches assignment table:  0->4, 1->5, 2->6, 3->7  (and back)")
    assert pos_n4 == [4, 5, 6, 7, 0, 1, 2, 3]

    print("\n[task4.4] NT-Xent on random projections:")
    for n in (4, 16, 64):
        z_fake = torch.randn(2 * n, 128, device=device)
        loss = nt_xent_loss(z_fake, temperature=TEMPERATURE).item()
        upper = math.log(2 * n - 1)
        print(
            f"  N={n:3d}  loss={loss:.4f}  "
            f"random-floor ~ log(2N-1) = {upper:.4f}"
        )

    print("\n[task4] sanity check passed. Ready for Task 5 pretraining.")


def simclr_train_one_epoch(model: nn.Module,loader: DataLoader,optimizer: torch.optim.Optimizer,device: torch.device,temperature: float,) -> float:
    model.train()   
    total_loss = 0.0
    total_examples = 0
    for (v1, v2), _label in loader:
        x = torch.cat([v1, v2], dim=0).to(device, non_blocking=True)
        z = model(x)
        loss = nt_xent_loss(z, temperature=temperature)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = v1.size(0)
        total_loss += loss.item() * bs
        total_examples += bs
    return total_loss / max(total_examples, 1)


def compute_post_training_similarity(model: SimCLRModel,device: torch.device,batch_size: int = 64,) -> Dict[str, float]:
 
    simclr_t = build_simclr_transform()
    two_view = TwoViewTransform(simclr_t)
    indices = load_indices("val")[:batch_size]
    dataset = IndexedSubset(get_cifar10_train(), indices, transform=two_view)
    loader = build_loader(dataset, batch_size=batch_size, shuffle=False)

    encoder = model.encoder
    encoder.eval()
    (v1, v2), _ = next(iter(loader))
    x = torch.cat([v1, v2], dim=0).to(device)
    with torch.no_grad():
        h = encoder(x)  # (2N, 512)

    same = mean_same_image_similarity(h, batch_size)
    diff = mean_different_image_similarity(h, batch_size)

    S = cosine_similarity_matrix(h)
    plot_similarity_heatmap(
        S,
        out_path=SIM_MATRIX_AFTER_PLOT,
        title=f"Cosine similarity (SimCLR encoder, N={batch_size})",
        batch_size=batch_size,
    )
    return {
        "same_view_similarity_after": float(same),
        "different_image_similarity_after": float(diff),
        "gap_after": float(same - diff),
    }


def pretrain_simclr(epochs: int = EPOCHS,batch_size: int = BATCH_SIZE,learning_rate: float = LEARNING_RATE,temperature: float = TEMPERATURE,device: torch.device | None = None,) -> Dict:
    set_seed()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[task5] device={device}, epochs={epochs}, batch_size={batch_size}, "
        f"lr={learning_rate}, tau={temperature}"
    )

    two_view = TwoViewTransform(build_simclr_transform())
    dataset = build_ssl_dataset(two_view)
    loader = build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    print(f"[task5] dataset size = {len(dataset)}, batches/epoch = {len(loader)}")

    model = SimCLRModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses: List[float] = []
    wall_start = time.time()
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = simclr_train_one_epoch(model, loader, optimizer, device, temperature)
        epoch_losses.append(loss)
        print(
            f"[task5] epoch {epoch:02d}/{epochs} | loss={loss:.4f} | "
            f"time={time.time() - t0:.1f}s"
        )

    save_encoder(model.encoder, ENCODER_CKPT)
    print(f"[task5] saved {ENCODER_CKPT}")

    ensure_dir(Path(LOSS_PLOT).parent)
    plot_loss_curves(
        epoch_losses,
        epoch_losses,
        out_path=LOSS_PLOT,
        title=f"SimCLR pretraining loss ({epochs} epochs)",
    )
    print(f"[task5] saved {LOSS_PLOT}")

    sims = compute_post_training_similarity(model, device, batch_size=batch_size)
    print(
        f"[task5] same_view_similarity_after:       {sims['same_view_similarity_after']:.4f}"
    )
    print(
        f"[task5] different_image_similarity_after: {sims['different_image_similarity_after']:.4f}"
    )
    print(f"[task5] gap_after:                         {sims['gap_after']:+.4f}")

    metrics = {
        "epochs_run": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "temperature": temperature,
        "total_minutes": round((time.time() - wall_start) / 60.0, 2),
        "final_loss": float(epoch_losses[-1]),
        **sims,
    }
    Path(SIM_METRICS_PATH).write_text(json.dumps(metrics, indent=2))
    print(f"[task5] saved {SIM_METRICS_PATH}")
    return {"epoch_losses": epoch_losses, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="SimCLR (Tasks 4 + 5)")
    parser.add_argument(
        "--train", action="store_true",
        help="Run the 50-epoch SimCLR pretraining (Task 5). "
             "Without this flag, only the Task 4 sanity check runs.",
    )
    args = parser.parse_args()

    ensure_dir("graphs")
    ensure_dir("results")
    ensure_dir("models")

    if args.train:
        pretrain_simclr()
    else:
        task4_sanity_check()


if __name__ == "__main__":
    main()
