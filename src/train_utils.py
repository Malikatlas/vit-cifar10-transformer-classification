"""Generic training utilities for all CIFAR-10 assignment models."""

from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm


def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Use CUDA when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate of the first optimizer group."""
    return float(optimizer.param_groups[0]["lr"])


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).sum().item())


def _peak_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / (1024**2)


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    model_name: str = "model",
) -> dict[str, float]:
    """Train a model for one epoch and return loss/accuracy/timing stats."""
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    running_loss = 0.0
    running_correct = 0.0
    running_total = 0
    start_time = time.perf_counter()

    progress = tqdm(dataloader, desc=f"{model_name} train epoch {epoch}", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += _accuracy_from_logits(logits, labels)
        running_total += batch_size

        progress.set_postfix(
            loss=running_loss / max(running_total, 1),
            acc=running_correct / max(running_total, 1),
        )

    epoch_time = time.perf_counter() - start_time
    return {
        "train_loss": running_loss / max(running_total, 1),
        "train_accuracy": running_correct / max(running_total, 1),
        "epoch_time": epoch_time,
        "peak_gpu_memory_mb": _peak_gpu_memory_mb(device),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str = "model",
) -> dict[str, float]:
    """Evaluate a model on a validation or test dataloader."""
    model.eval()
    running_loss = 0.0
    running_correct = 0.0
    running_total = 0

    progress = tqdm(dataloader, desc=f"{model_name} eval", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += _accuracy_from_logits(logits, labels)
        running_total += batch_size

    return {
        "val_loss": running_loss / max(running_total, 1),
        "val_accuracy": running_correct / max(running_total, 1),
    }


def save_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    """Save per-epoch training history to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_history_json(history: list[dict[str, Any]], path: Path) -> None:
    """Save per-epoch training history to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def fit_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    model_name: str,
    checkpoint_path: str | Path,
    scheduler: torch.optim.lr_scheduler.LRScheduler | ReduceLROnPlateau | None = None,
    early_stopping_patience: int = 8,
    min_delta: float = 1e-4,
    history_csv_path: str | Path | None = None,
    history_json_path: str | Path | None = None,
    extra_checkpoint: dict[str, Any] | None = None,
    initial_best_val_loss: float | None = None,
    initial_best_val_accuracy: float = 0.0,
) -> dict[str, Any]:
    """Train with validation, scheduler updates, early stopping, and checkpointing."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model.to(device)
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf") if initial_best_val_loss is None else initial_best_val_loss
    best_val_accuracy = initial_best_val_accuracy
    epochs_without_improvement = 0
    total_start_time = time.perf_counter()

    print(f"\nTraining {model_name} for {epochs} epoch(s) on {device}.")
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, model_name)
        val_stats = evaluate(model, val_loader, criterion, device, model_name)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_stats["val_loss"])
        elif scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "train_accuracy": train_stats["train_accuracy"],
            "val_loss": val_stats["val_loss"],
            "val_accuracy": val_stats["val_accuracy"],
            "epoch_time": train_stats["epoch_time"],
            "learning_rate": get_learning_rate(optimizer),
            "peak_gpu_memory_mb": train_stats["peak_gpu_memory_mb"],
        }
        history.append(row)

        print(
            f"{model_name} epoch {epoch:03d}: "
            f"train_loss={row['train_loss']:.4f}, train_acc={row['train_accuracy']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, val_acc={row['val_accuracy']:.4f}, "
            f"lr={row['learning_rate']:.2e}, time={row['epoch_time']:.1f}s, "
            f"peak_gpu={row['peak_gpu_memory_mb']:.1f} MB"
        )

        improved = row["val_loss"] < best_val_loss - min_delta
        if improved:
            best_val_loss = row["val_loss"]
            best_val_accuracy = row["val_accuracy"]
            epochs_without_improvement = 0
            checkpoint = {
                "model_name": model_name,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_accuracy,
                "history": history,
            }
            if extra_checkpoint:
                checkpoint.update(extra_checkpoint)
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping {model_name} after {epoch} epochs.")
                break

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

    if history_csv_path is not None:
        save_history_csv(history, Path(history_csv_path))
    if history_json_path is not None:
        save_history_json(history, Path(history_json_path))

    total_training_time = time.perf_counter() - total_start_time
    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "checkpoint_path": str(checkpoint_path),
        "total_training_time": total_training_time,
        "average_epoch_time": float(np.mean([row["epoch_time"] for row in history])) if history else 0.0,
        "peak_gpu_memory_mb": max((row["peak_gpu_memory_mb"] for row in history), default=0.0),
    }
