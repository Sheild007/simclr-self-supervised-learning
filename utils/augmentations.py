from __future__ import annotations
import torchvision.transforms as T


CIFAR10_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def build_supervised_train_transform() -> T.Compose:  
    return T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def build_eval_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


__all__ = [
    "CIFAR10_MEAN",
    "CIFAR10_STD",
    "build_supervised_train_transform",
    "build_eval_transform",
]
