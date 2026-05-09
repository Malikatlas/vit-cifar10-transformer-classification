"""Evaluate saved CIFAR-10 assignment checkpoints on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from src.config import DEFAULT_HYBRID_CONFIG, DEFAULT_VIT_CONFIG, NUM_WORKERS, RANDOM_SEED, ensure_output_dirs, resolve_data_root, resolve_output_dir
from src.data import get_cifar10_dataloaders
from src.eval_utils import evaluate_classifier, save_classification_outputs, save_comparison_markdown, save_metrics_summary
from src.models import HybridCNNMLP, VisionTransformer
from src.models.resnet_transfer import create_resnet18_transfer
from src.train_utils import count_parameters, get_device, set_seed
from src.visualization import plot_confusion_matrix, plot_training_curves, save_correct_incorrect_grids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all saved CIFAR-10 assignment models.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resnet-batch-size", type=int, default=64)
    parser.add_argument("--vit-checkpoint", type=str, default="outputs/checkpoints/vit_best.pt")
    parser.add_argument("--hybrid-checkpoint", type=str, default="outputs/checkpoints/hybrid_best.pt")
    parser.add_argument("--resnet-checkpoint", type=str, default="outputs/checkpoints/resnet_best.pt")
    return parser.parse_args()


def _checkpoint_path(path_text: str, project_root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def _build_model(model_key: str, checkpoint: dict) -> nn.Module:
    if model_key == "vit":
        config = DEFAULT_VIT_CONFIG.copy()
        config.update(checkpoint.get("model_config", {}))
        return VisionTransformer(**config)
    if model_key == "hybrid":
        config = DEFAULT_HYBRID_CONFIG.copy()
        config.update(checkpoint.get("model_config", {}))
        return HybridCNNMLP(**config)
    if model_key == "resnet":
        config = checkpoint.get("model_config", {"num_classes": 10})
        return create_resnet18_transfer(
            num_classes=config.get("num_classes", 10),
            pretrained=False,
            freeze_backbone=False,
        )
    raise ValueError(f"Unknown model key: {model_key}")


def _summary_row(display_name: str, checkpoint: dict, eval_result: dict, model: nn.Module) -> dict:
    total_parameters, trainable_parameters = count_parameters(model)
    history = checkpoint.get("history", [])
    total_training_time = sum(row.get("epoch_time", 0.0) for row in history)
    average_epoch_time = total_training_time / max(len(history), 1)
    peak_gpu_memory_mb = max((row.get("peak_gpu_memory_mb", 0.0) for row in history), default=0.0)
    return {
        "model_name": display_name,
        "best_validation_accuracy": checkpoint.get("best_val_accuracy", 0.0),
        "test_accuracy": eval_result["accuracy"],
        "precision": eval_result["precision"],
        "recall": eval_result["recall"],
        "f1_score": eval_result["f1_score"],
        "total_training_time": total_training_time,
        "average_epoch_time": average_epoch_time,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "inference_images_per_second": eval_result["inference_images_per_second"],
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)
    project_root = Path(__file__).resolve().parent
    data_root = resolve_data_root(args.data_root)
    output_dir = resolve_output_dir(args.output_dir)
    ensure_output_dirs(output_dir)
    device = get_device()

    checkpoint_specs = {
        "vit": _checkpoint_path(args.vit_checkpoint, project_root),
        "hybrid": _checkpoint_path(args.hybrid_checkpoint, project_root),
        "resnet": _checkpoint_path(args.resnet_checkpoint, project_root),
    }
    missing = [str(path) for path in checkpoint_specs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoint files:\n - " + "\n - ".join(missing))

    standard_loaders = get_cifar10_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        model_family="standard",
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        smoke_test=False,
    )
    resnet_loaders = get_cifar10_dataloaders(
        data_root=data_root,
        batch_size=args.resnet_batch_size,
        model_family="resnet",
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        smoke_test=False,
    )
    class_names = standard_loaders["class_names"]

    model_specs = [
        ("vit", "ViT from scratch", "standard", standard_loaders["test"]),
        ("hybrid", "Hybrid CNN + MLP", "standard", standard_loaders["test"]),
        ("resnet", "Pretrained ResNet18 transfer learning", "resnet", resnet_loaders["test"]),
    ]

    summary_rows: list[dict] = []
    for model_key, display_name, model_family, test_loader in model_specs:
        checkpoint = _load_checkpoint(checkpoint_specs[model_key], device)
        model = _build_model(model_key, checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        history = checkpoint.get("history", [])
        if history:
            plot_training_curves(history, model_key, output_dir)

        eval_result = evaluate_classifier(
            model,
            test_loader,
            device,
            class_names,
            model_name=model_key,
            collect_images=True,
        )
        save_classification_outputs(eval_result, output_dir, model_key)
        plot_confusion_matrix(eval_result["confusion_matrix"], class_names, model_key, output_dir)
        save_correct_incorrect_grids(eval_result, class_names, model_key, output_dir, model_family=model_family)
        summary_rows.append(_summary_row(display_name, checkpoint, eval_result, model))
        print(f"{display_name}: test_accuracy={eval_result['accuracy']:.4f}, f1={eval_result['f1_score']:.4f}")

    save_metrics_summary(summary_rows, output_dir)
    save_comparison_markdown(summary_rows, output_dir)
    print(f"\nSaved evaluation outputs under {output_dir}")


if __name__ == "__main__":
    main()

