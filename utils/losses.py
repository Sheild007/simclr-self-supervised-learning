from __future__ import annotations
import torch
import torch.nn.functional as F


DEFAULT_TEMPERATURE: float = 0.5


def cosine_similarity_matrix(features: torch.Tensor) -> torch.Tensor:
    z = F.normalize(features, dim=1)
    return z @ z.t()


def mean_same_image_similarity(features: torch.Tensor, batch_size: int) -> float:
    S = cosine_similarity_matrix(features)
    n = batch_size
    rows = torch.arange(n, device=S.device)
    cols = rows + n
    return S[rows, cols].mean().item()


def mean_different_image_similarity( features: torch.Tensor, batch_size: int) -> float:
    S = cosine_similarity_matrix(features)
    two_n = S.size(0)
    n = batch_size

    mask = torch.ones_like(S, dtype=torch.bool)
    diag = torch.arange(two_n, device=S.device)
    mask[diag, diag] = False
    rows = torch.arange(n, device=S.device)
    mask[rows, rows + n] = False
    mask[rows + n, rows] = False

    return S[mask].mean().item()


def build_positive_indices(batch_size: int) -> torch.Tensor:
    n = batch_size
    idx = torch.arange(2 * n)
    pos = torch.where(idx < n, idx + n, idx - n)
    return pos.long()


def nt_xent_loss(features: torch.Tensor,temperature: float = DEFAULT_TEMPERATURE,) -> torch.Tensor:
    if features.dim() != 2:
        raise ValueError(f"features must be 2D, got shape {tuple(features.shape)}")
    if features.size(0) % 2 != 0:
        raise ValueError("features.size(0) must be 2N (even).")

    two_n = features.size(0)
    n = two_n // 2
    device = features.device

    z = F.normalize(features, dim=1)
    logits = (z @ z.t()) / temperature

    mask_self = torch.eye(two_n, dtype=torch.bool, device=device)
    logits = logits.masked_fill(mask_self, float("-inf"))

    pos_idx = build_positive_indices(n).to(device)
    return F.cross_entropy(logits, pos_idx)


__all__ = [
    "DEFAULT_TEMPERATURE",
    "cosine_similarity_matrix",
    "mean_same_image_similarity",
    "mean_different_image_similarity",
    "build_positive_indices",
    "nt_xent_loss",
]
