from __future__ import annotations
import torch
import torch.nn.functional as F


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


__all__ = [
    "cosine_similarity_matrix",
    "mean_same_image_similarity",
    "mean_different_image_similarity",
]
