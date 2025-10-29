from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Callable

import os
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset

def dl_worker_init(_wid: int) -> None:
    torch.set_num_threads(1)

@dataclass(frozen=True)
class LoaderPolicy:
    pin_memory: bool
    num_workers: int
    persistent_workers: bool
    worker_init_fn: Optional[Callable[[int], None]]

def resolve_loader_env(
    device: torch.device,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> LoaderPolicy:
    is_windows = os.name == "nt"
    if num_workers is None:
        cpu_n = os.cpu_count() or 2
        nw = 2 if is_windows else min(4, cpu_n)
    else:
        nw = int(num_workers)
    if is_windows and nw > 2:
        nw = 2
    pin = (device.type == "cuda" and not is_windows) if pin_memory is None else (bool(pin_memory) and not is_windows)
    persistent = nw > 0 and not is_windows
    worker_init = dl_worker_init if nw > 0 else None
    return LoaderPolicy(pin_memory=pin, num_workers=nw, persistent_workers=persistent, worker_init_fn=worker_init)

def make_loader(
    ds,
    indices: Optional[Sequence[int]],
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, bool]:
    policy = resolve_loader_env(device, num_workers=num_workers, pin_memory=pin_memory)
    data = Subset(ds, indices) if indices is not None else ds
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=policy.pin_memory,
        num_workers=policy.num_workers,
        persistent_workers=policy.persistent_workers,
        worker_init_fn=policy.worker_init_fn,
    )
    return loader, policy.pin_memory

NORMS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashionmnist": ((0.1307,), (0.3081,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

class IndexedDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y = self.base[i]
        return x, y, i

def get_dataset(dataset_name: str, want_channels: int = 1) -> Tuple[Dataset, Dataset]:
    ds = (dataset_name or "MNIST").strip().lower()
    tfms = []
    if ds == "cifar10" and int(want_channels) == 1:
        tfms.append(transforms.Grayscale(num_output_channels=1))
    tfms.append(transforms.ToTensor())
    tfm = transforms.Compose(tfms)
    root = "./data"
    if ds == "mnist":
        tr = datasets.MNIST(root, train=True, download=True, transform=tfm)
        te = datasets.MNIST(root, train=False, download=True, transform=tfm)
    elif ds == "fashionmnist":
        tr = datasets.FashionMNIST(root, train=True, download=True, transform=tfm)
        te = datasets.FashionMNIST(root, train=False, download=True, transform=tfm)
    elif ds == "cifar10":
        tr = datasets.CIFAR10(root, train=True, download=True, transform=tfm)
        te = datasets.CIFAR10(root, train=False, download=True, transform=tfm)
    else:
        raise ValueError("dataset must be MNIST|FashionMNIST|CIFAR10")
    return tr, te

def _normalize(base: transforms.Compose, channels: int, dataset: str, enable: bool) -> transforms.Compose:
    if not enable:
        return base
    pair = NORMS.get((dataset or "").lower())
    if not pair:
        return base
    mean, std = pair
    if len(mean) != channels:
        if channels == 1:
            mean = (float(sum(mean) / len(mean)),)
            std = (float(sum(std) / len(std)),)
        else:
            m = mean[0] if len(mean) == 1 else float(sum(mean) / len(mean))
            s = std[0] if len(std) == 1 else float(sum(std) / len(std))
            mean = tuple([m] * channels)
            std = tuple([s] * channels)
    return transforms.Compose([base, transforms.Normalize(mean, std)])

def _expected_channels(model_name: str, dataset_name: str) -> int:
    mm = (model_name or "").strip().lower()
    dd = (dataset_name or "").strip().lower()
    if mm in {"resnet8", "resnet16"}:
        return 3 if dd == "cifar10" else 1
    if mm == "mojocifarcnn":
        return 3
    return 1

def get_dataset_and_transform(dataset_name: str, model_name: str, normalize: bool = True) -> Tuple[VisionDataset, VisionDataset]:
    ds = (dataset_name or "").strip().lower()
    if ds not in {"mnist", "fashionmnist", "cifar10"}:
        raise ValueError("dataset must be MNIST|FashionMNIST|CIFAR10")
    final_c = _expected_channels(model_name, dataset_name)
    dataset_c = 3 if ds == "cifar10" else 1
    tfms = []
    if dataset_c != final_c:
        tfms.append(transforms.Grayscale(num_output_channels=final_c))
    tfms.append(transforms.ToTensor())
    base = transforms.Compose(tfms)
    composed = _normalize(base, final_c, ds, normalize)
    root = "./data"
    if ds == "mnist":
        train_ds = datasets.MNIST(root, train=True, download=True, transform=composed)
        test_ds = datasets.MNIST(root, train=False, transform=composed)
    elif ds == "fashionmnist":
        train_ds = datasets.FashionMNIST(root, train=True, download=True, transform=composed)
        test_ds = datasets.FashionMNIST(root, train=False, transform=composed)
    else:
        train_ds = datasets.CIFAR10(root, train=True, download=True, transform=composed)
        test_ds = datasets.CIFAR10(root, train=False, transform=composed)
    return train_ds, test_ds
