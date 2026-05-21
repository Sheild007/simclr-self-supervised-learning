from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18

ENCODER_DIM: int = 512 


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


__all__ = ["ENCODER_DIM", "CIFARResNet18", "ClassifierModel"]
