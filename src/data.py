"""Data loading and preprocessing for CIFAR-10.

All loaders use the local CIFAR-10 Python dataset through
``torchvision.datasets.CIFAR10(..., download=False)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import (
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    DATA_ROOT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    RANDOM_SEED,
)


ModelFamily = Literal["standard", "resnet"]


def validate_cifar10_root(data_root: str | Path = DATA_ROOT) -> Path:
    """Validate that the extracted CIFAR-10 Python batch folder exists."""
    root = Path(data_root).expanduser().resolve()
    batch_dir = root / "cifar-10-batches-py"
    if not batch_dir.exists():
        raise FileNotFoundError(
            "CIFAR-10 batch folder was not found. Expected: "
            f"{batch_dir}. Place cifar-10-batches-py under {root} and run with "
            "download=False."
        )
    return root


def get_standard_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Transforms for the from-scratch ViT and Hybrid CNN + MLP models."""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, eval_transform


def get_resnet_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Transforms for ImageNet-pretrained ResNet18 transfer learning."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def get_transforms(model_family: ModelFamily = "standard") -> tuple[transforms.Compose, transforms.Compose]:
    """Return train/evaluation transforms for a model family."""
    if model_family == "resnet":
        return get_resnet_transforms()
    return get_standard_transforms()


def _split_train_val_indices(
    dataset_size: int = 50_000,
    val_size: int = 5_000,
    seed: int = RANDOM_SEED,
) -> tuple[list[int], list[int]]:
    """Create a fixed 45k/5k train/validation split."""
    generator = torch.Generator().manual_seed(seed)
    shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()
    val_indices = shuffled_indices[:val_size]
    train_indices = shuffled_indices[val_size:]
    return train_indices, val_indices


def _limit_indices(indices: list[int], limit: int | None) -> list[int]:
    return indices if limit is None else indices[:limit]


def get_cifar10_dataloaders(
    data_root: str | Path = DATA_ROOT,
    batch_size: int = 128,
    model_family: ModelFamily = "standard",
    num_workers: int = NUM_WORKERS,
    seed: int = RANDOM_SEED,
    smoke_test: bool = False,
    train_subset_size: int | None = None,
    val_subset_size: int | None = None,
    test_subset_size: int | None = None,
) -> dict[str, object]:
    """Build CIFAR-10 train/validation/test dataloaders.

    The training split is 45,000 images and the validation split is 5,000
    images. Smoke-test mode keeps the exact same split logic but limits the
    number of samples to make a full end-to-end run quick.
    """
    root = validate_cifar10_root(data_root)
    train_transform, eval_transform = get_transforms(model_family)

    train_full = datasets.CIFAR10(root=root, train=True, transform=train_transform, download=False)
    val_full = datasets.CIFAR10(root=root, train=True, transform=eval_transform, download=False)
    test_full = datasets.CIFAR10(root=root, train=False, transform=eval_transform, download=False)

    train_indices, val_indices = _split_train_val_indices(len(train_full), seed=seed)

    if smoke_test:
        train_subset_size = 1024 if train_subset_size is None else train_subset_size
        val_subset_size = 256 if val_subset_size is None else val_subset_size
        test_subset_size = 256 if test_subset_size is None else test_subset_size

    train_indices = _limit_indices(train_indices, train_subset_size)
    val_indices = _limit_indices(val_indices, val_subset_size)

    test_indices = list(range(len(test_full)))
    if test_subset_size is not None:
        generator = torch.Generator().manual_seed(seed)
        test_indices = torch.randperm(len(test_full), generator=generator).tolist()[:test_subset_size]

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)
    test_dataset = Subset(test_full, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "class_names": CIFAR10_CLASSES,
    }


def get_raw_cifar10_dataset(data_root: str | Path = DATA_ROOT, train: bool = True) -> datasets.CIFAR10:
    """Return a raw PIL-image CIFAR-10 dataset for visualization."""
    root = validate_cifar10_root(data_root)
    return datasets.CIFAR10(root=root, train=train, transform=None, download=False)

