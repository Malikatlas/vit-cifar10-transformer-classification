"""Visualization utilities for CIFAR-10 training and evaluation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import CIFAR10_MEAN, CIFAR10_STD, IMAGENET_MEAN, IMAGENET_STD
from .data import get_raw_cifar10_dataset, get_standard_transforms


def _to_numpy_image(tensor: torch.Tensor, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    """Unnormalize a CHW tensor and return an HWC NumPy image."""
    mean_tensor = torch.tensor(list(mean)).view(3, 1, 1)
    std_tensor = torch.tensor(list(std)).view(3, 1, 1)
    image = tensor.detach().cpu() * std_tensor + mean_tensor
    image = image.clamp(0, 1).permute(1, 2, 0).numpy()
    return image


def save_cifar10_samples(
    data_root: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    num_images: int = 25,
) -> Path:
    """Save a grid of original, unaugmented CIFAR-10 samples."""
    dataset = get_raw_cifar10_dataset(data_root, train=True)
    output_path = Path(output_dir) / "plots" / "cifar10_samples.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid_size = int(np.ceil(np.sqrt(num_images)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2.0, grid_size * 2.0))
    axes = np.asarray(axes).reshape(-1)

    for axis, index in zip(axes, range(num_images)):
        image, label = dataset[index]
        axis.imshow(image)
        axis.set_title(class_names[label], fontsize=9)
        axis.axis("off")
    for axis in axes[num_images:]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_augmentation_visualization(
    data_root: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    num_images: int = 8,
) -> Path:
    """Save original and augmented versions of CIFAR-10 samples."""
    raw_dataset = get_raw_cifar10_dataset(data_root, train=True)
    train_transform, _ = get_standard_transforms()
    output_path = Path(output_dir) / "plots" / "cifar10_augmentations.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 1.7, 3.8))
    for column in range(num_images):
        image, label = raw_dataset[column]
        augmented = train_transform(image)

        axes[0, column].imshow(image)
        axes[0, column].set_title(class_names[label], fontsize=8)
        axes[0, column].axis("off")

        axes[1, column].imshow(_to_numpy_image(augmented, CIFAR10_MEAN, CIFAR10_STD))
        axes[1, column].set_title("augmented", fontsize=8)
        axes[1, column].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_training_curves(history: list[dict], model_name: str, output_dir: str | Path) -> tuple[Path, Path]:
    """Save training/validation accuracy and loss curves for a model."""
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = [row["epoch"] for row in history]

    acc_path = plots_dir / f"{model_name}_accuracy_curve.png"
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, [row["train_accuracy"] for row in history], label="Train")
    axis.plot(epochs, [row["val_accuracy"] for row in history], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title(f"{model_name} accuracy")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(acc_path, dpi=160)
    plt.close(fig)

    loss_path = plots_dir / f"{model_name}_loss_curve.png"
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(epochs, [row["train_loss"] for row in history], label="Train")
    axis.plot(epochs, [row["val_loss"] for row in history], label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title(f"{model_name} loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(loss_path, dpi=160)
    plt.close(fig)
    return acc_path, loss_path


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_dir: str | Path,
) -> Path:
    """Save a confusion matrix image."""
    output_path = Path(output_dir) / "plots" / f"{model_name}_confusion_matrix.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"{model_name} confusion matrix")

    threshold = confusion.max() / 2.0 if confusion.size else 0
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            color = "white" if confusion[row, column] > threshold else "black"
            axis.text(column, row, int(confusion[row, column]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_prediction_grid(
    images: torch.Tensor,
    true_labels: np.ndarray | torch.Tensor,
    predictions: np.ndarray | torch.Tensor,
    probabilities: np.ndarray | torch.Tensor,
    class_names: list[str],
    output_path: str | Path,
    title: str,
    mean: Iterable[float] = CIFAR10_MEAN,
    std: Iterable[float] = CIFAR10_STD,
    max_images: int = 16,
) -> Path:
    """Save a grid of predicted images with true/predicted labels."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if torch.is_tensor(true_labels):
        true_labels = true_labels.detach().cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(probabilities):
        probabilities = probabilities.detach().cpu().numpy()

    images = images.detach().cpu()[:max_images]
    true_labels = np.asarray(true_labels)[: len(images)]
    predictions = np.asarray(predictions)[: len(images)]
    probabilities = np.asarray(probabilities)[: len(images)]

    if len(images) == 0:
        return output_path

    grid_columns = min(4, len(images))
    grid_rows = int(np.ceil(len(images) / grid_columns))
    fig, axes = plt.subplots(grid_rows, grid_columns, figsize=(grid_columns * 3.0, grid_rows * 3.0))
    axes = np.asarray(axes).reshape(-1)

    for index, axis in enumerate(axes):
        if index >= len(images):
            axis.axis("off")
            continue
        image = _to_numpy_image(images[index], mean, std)
        confidence = float(probabilities[index, predictions[index]])
        axis.imshow(image)
        axis.set_title(
            f"T: {class_names[int(true_labels[index])]}\n"
            f"P: {class_names[int(predictions[index])]} ({confidence:.2f})",
            fontsize=9,
        )
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def save_correct_incorrect_grids(
    result: dict,
    class_names: list[str],
    model_name: str,
    output_dir: str | Path,
    model_family: str = "standard",
    max_images: int = 16,
) -> tuple[Path | None, Path | None]:
    """Save correct and incorrect prediction grids from collected evaluation images."""
    images = result.get("images")
    if images is None:
        return None, None

    true_labels = result["true_labels"][: len(images)]
    predictions = result["predictions"][: len(images)]
    probabilities = result["probabilities"][: len(images)]
    correct_mask = predictions == true_labels
    correct_mask_tensor = torch.as_tensor(correct_mask, dtype=torch.bool)

    if model_family == "resnet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD

    output_dir = Path(output_dir)
    correct_path = None
    incorrect_path = None

    if np.any(correct_mask):
        correct_path = save_prediction_grid(
            images[correct_mask_tensor],
            true_labels[correct_mask],
            predictions[correct_mask],
            probabilities[correct_mask],
            class_names,
            output_dir / "plots" / f"{model_name}_correct_predictions.png",
            f"{model_name} correct predictions",
            mean,
            std,
            max_images,
        )

    if np.any(~correct_mask):
        incorrect_path = save_prediction_grid(
            images[~correct_mask_tensor],
            true_labels[~correct_mask],
            predictions[~correct_mask],
            probabilities[~correct_mask],
            class_names,
            output_dir / "plots" / f"{model_name}_incorrect_predictions.png",
            f"{model_name} incorrect predictions",
            mean,
            std,
            max_images,
        )

    return correct_path, incorrect_path
