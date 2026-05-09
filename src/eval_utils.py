"""Evaluation helpers for CIFAR-10 classifiers."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from tqdm.auto import tqdm


def _json_ready(value: Any) -> Any:
    """Convert NumPy/PyTorch values to JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: list[str],
    model_name: str = "model",
    criterion: nn.Module | None = None,
    collect_images: bool = False,
    max_collect_images: int = 512,
) -> dict[str, Any]:
    """Evaluate a classifier and collect predictions, probabilities, and labels."""
    model.eval()
    criterion = criterion or nn.CrossEntropyLoss()

    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    collected_images: list[torch.Tensor] = []

    running_loss = 0.0
    running_total = 0
    start_time = time.perf_counter()

    progress = tqdm(dataloader, desc=f"{model_name} test", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = F.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_total += batch_size

        all_targets.append(labels.detach().cpu().numpy())
        all_predictions.append(predictions.detach().cpu().numpy())
        all_probabilities.append(probabilities.detach().cpu().numpy())

        if collect_images and len(collected_images) < max_collect_images:
            remaining = max_collect_images - len(collected_images)
            collected_images.extend(images.detach().cpu()[:remaining])

    elapsed = max(time.perf_counter() - start_time, 1e-9)
    targets = np.concatenate(all_targets)
    predictions = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)

    report_dict = classification_report(
        targets,
        predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        targets,
        predictions,
        target_names=class_names,
        zero_division=0,
    )

    result = {
        "test_loss": running_loss / max(running_total, 1),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, average="macro", zero_division=0),
        "recall": recall_score(targets, predictions, average="macro", zero_division=0),
        "f1_score": f1_score(targets, predictions, average="macro", zero_division=0),
        "classification_report": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": confusion_matrix(targets, predictions),
        "inference_images_per_second": running_total / elapsed,
        "predictions": predictions,
        "true_labels": targets,
        "probabilities": probabilities,
        "images": torch.stack(collected_images) if collected_images else None,
    }
    return result


def save_classification_outputs(result: dict[str, Any], output_dir: str | Path, model_name: str) -> None:
    """Save classification report and raw evaluation arrays for one model."""
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    report_json_path = metrics_dir / f"{model_name}_classification_report.json"
    report_txt_path = metrics_dir / f"{model_name}_classification_report.txt"
    arrays_path = metrics_dir / f"{model_name}_predictions.npz"

    with report_json_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(result["classification_report"]), file, indent=2)
    with report_txt_path.open("w", encoding="utf-8") as file:
        file.write(result["classification_report_text"])

    np.savez_compressed(
        arrays_path,
        predictions=result["predictions"],
        true_labels=result["true_labels"],
        probabilities=result["probabilities"],
        confusion_matrix=result["confusion_matrix"],
    )


def save_metrics_summary(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Save the final model comparison as CSV and JSON."""
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = metrics_dir / "metrics_summary.csv"
    json_path = metrics_dir / "metrics_summary.json"

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(_json_ready(rows), file, indent=2)


def save_comparison_markdown(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Save a simple Markdown comparison table for reports/notebooks."""
    if not rows:
        return
    output_dir = Path(output_dir)
    path = output_dir / "metrics" / "comparison_table.md"
    path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "model_name",
        "best_validation_accuracy",
        "test_accuracy",
        "precision",
        "recall",
        "f1_score",
        "total_training_time",
        "peak_gpu_memory_mb",
        "inference_images_per_second",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.4f}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

