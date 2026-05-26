from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18

ENCODER_DIM: int = 512
PROJECTION_HIDDEN_DIM: int = 256
PROJECTION_OUT_DIM: int = 128


class CIFARResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False,)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ClassifierModel(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = CIFARResNet18()
        self.classifier = nn.Linear(ENCODER_DIM, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()


class ProjectionHead(nn.Module):

    def __init__(self,in_dim: int = ENCODER_DIM,hidden_dim: int = PROJECTION_HIDDEN_DIM,out_dim: int = PROJECTION_OUT_DIM,) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SimCLRModel(nn.Module):
  
    def __init__(self,encoder: CIFARResNet18 | None = None,projection_head: ProjectionHead | None = None,) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else CIFARResNet18()
        self.projection_head = (
            projection_head if projection_head is not None else ProjectionHead()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        z = self.projection_head(h)
        return z


def save_encoder(encoder: nn.Module, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    target = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
    torch.save(target.state_dict(), path)


def load_encoder_weights(encoder: nn.Module,path: str,map_location: str | torch.device | None = None,) -> None:
    state_dict = torch.load(path, map_location=map_location)
    target = encoder.module if isinstance(encoder, nn.DataParallel) else encoder
    target.load_state_dict(state_dict)


__all__ = [
    "ENCODER_DIM",
    "PROJECTION_HIDDEN_DIM",
    "PROJECTION_OUT_DIM",
    "CIFARResNet18",
    "ClassifierModel",
    "ProjectionHead",
    "SimCLRModel",
    "save_encoder",
    "load_encoder_weights",
]
