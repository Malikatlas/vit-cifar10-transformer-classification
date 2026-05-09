"""Train and compare ViT, Hybrid CNN + MLP, and ResNet18 on CIFAR-10."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from src.config import (
    DEFAULT_HYBRID_CONFIG,
    DEFAULT_VIT_CONFIG,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    RANDOM_SEED,
    ensure_output_dirs,
    resolve_data_root,
    resolve_output_dir,
)
from src.data import get_cifar10_dataloaders
from src.eval_utils import evaluate_classifier, save_classification_outputs, save_comparison_markdown, save_metrics_summary
from src.models import HybridCNNMLP, VisionTransformer
from src.models.resnet_transfer import create_resnet18_transfer, unfreeze_layer4
from src.train_utils import (
    count_parameters,
    fit_model,
    get_device,
    save_history_csv,
    save_history_json,
    set_seed,
)
from src.visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    save_augmentation_visualization,
    save_cifar10_samples,
    save_correct_incorrect_grids,
)


VIT_TUNING_CONFIGS = [
    {"patch_size": 4, "depth": 4, "num_heads": 4, "learning_rate": 3e-4},
    {"patch_size": 4, "depth": 6, "num_heads": 6, "learning_rate": 3e-4},
    {"patch_size": 8, "depth": 6, "num_heads": 6, "learning_rate": 3e-4},
    {"patch_size": 4, "depth": 6, "num_heads": 6, "learning_rate": 1e-4},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all CIFAR-10 assignment models.")
    parser.add_argument("--smoke-test", action="store_true", help="Use tiny subsets and one epoch per model.")
    parser.add_argument("--tune-vit-fast", action="store_true", help="Run lightweight ViT hyperparameter tuning.")
    parser.add_argument("--epochs-vit", type=int, default=30)
    parser.add_argument("--epochs-hybrid", type=int, default=25)
    parser.add_argument("--epochs-resnet", type=int, default=10)
    parser.add_argument("--epochs-resnet-finetune", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--resnet-batch-size", type=int, default=64)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-resnet-finetune", action="store_true")
    return parser.parse_args()


def _save_tuning_results(rows: list[dict], output_dir: Path) -> None:
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "vit_hyperparameter_tuning.csv"
    if not rows:
        return
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved ViT tuning results: {csv_path}")


def tune_vit_fast(args: argparse.Namespace, data_root: Path, output_dir: Path, device: torch.device) -> dict:
    """Run a small ViT hyperparameter search and return the best config."""
    print("\nStarting lightweight ViT hyperparameter tuning.")
    rows: list[dict] = []
    best_config = DEFAULT_VIT_CONFIG.copy()
    best_accuracy = -1.0
    tuning_epochs = 1 if args.smoke_test else 3

    for index, tuning_config in enumerate(VIT_TUNING_CONFIGS, start=1):
        config = DEFAULT_VIT_CONFIG.copy()
        config.update(
            {
                "patch_size": tuning_config["patch_size"],
                "depth": tuning_config["depth"],
                "num_heads": tuning_config["num_heads"],
            }
        )
        learning_rate = tuning_config["learning_rate"]
        batch_size = tuning_config.get("batch_size", args.batch_size)

        loaders = get_cifar10_dataloaders(
            data_root=data_root,
            batch_size=batch_size,
            model_family="standard",
            num_workers=NUM_WORKERS,
            seed=RANDOM_SEED,
            smoke_test=True,
            train_subset_size=1024 if args.smoke_test else 2048,
            val_subset_size=256 if args.smoke_test else 512,
            test_subset_size=256,
        )
        model = VisionTransformer(**config)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
        criterion = nn.CrossEntropyLoss()
        scheduler = CosineAnnealingLR(optimizer, T_max=max(tuning_epochs, 1))
        start_time = time.perf_counter()
        result = fit_model(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=tuning_epochs,
            model_name=f"vit_tune_{index}",
            checkpoint_path=output_dir / "checkpoints" / f"vit_tune_{index}.pt",
            scheduler=scheduler,
            early_stopping_patience=2,
            extra_checkpoint={"model_type": "vit", "model_config": config},
        )
        elapsed = time.perf_counter() - start_time

        row = {
            "patch_size": config["patch_size"],
            "depth": config["depth"],
            "num_heads": config["num_heads"],
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "best_val_accuracy": result["best_val_accuracy"],
            "best_val_loss": result["best_val_loss"],
            "training_time": elapsed,
        }
        rows.append(row)

        if result["best_val_accuracy"] > best_accuracy:
            best_accuracy = result["best_val_accuracy"]
            best_config = config
            best_config["learning_rate"] = learning_rate

    _save_tuning_results(rows, output_dir)
    print(f"Recommended ViT config from tuning: {json.dumps(best_config, indent=2)}")
    return best_config


def _summary_row(display_name: str, train_result: dict, eval_result: dict, model: nn.Module) -> dict:
    total_parameters, trainable_parameters = count_parameters(model)
    return {
        "model_name": display_name,
        "best_validation_accuracy": train_result["best_val_accuracy"],
        "test_accuracy": eval_result["accuracy"],
        "precision": eval_result["precision"],
        "recall": eval_result["recall"],
        "f1_score": eval_result["f1_score"],
        "total_training_time": train_result["total_training_time"],
        "average_epoch_time": train_result["average_epoch_time"],
        "peak_gpu_memory_mb": train_result["peak_gpu_memory_mb"],
        "inference_images_per_second": eval_result["inference_images_per_second"],
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
    }


def _evaluate_and_save(
    slug: str,
    display_name: str,
    model: nn.Module,
    train_result: dict,
    test_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    output_dir: Path,
    device: torch.device,
    model_family: str = "standard",
) -> dict:
    eval_result = evaluate_classifier(
        model,
        test_loader,
        device,
        class_names,
        model_name=slug,
        collect_images=True,
    )
    save_classification_outputs(eval_result, output_dir, slug)
    plot_confusion_matrix(eval_result["confusion_matrix"], class_names, slug, output_dir)
    save_correct_incorrect_grids(eval_result, class_names, slug, output_dir, model_family=model_family)
    return _summary_row(display_name, train_result, eval_result, model)


def train_vit(args: argparse.Namespace, data_root: Path, output_dir: Path, device: torch.device, vit_config: dict) -> tuple[nn.Module, dict, dict]:
    loaders = get_cifar10_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        model_family="standard",
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        smoke_test=args.smoke_test,
    )
    config = DEFAULT_VIT_CONFIG.copy()
    config.update({key: value for key, value in vit_config.items() if key in DEFAULT_VIT_CONFIG})
    learning_rate = vit_config.get("learning_rate", 3e-4)

    epochs = 1 if args.smoke_test else args.epochs_vit
    model = VisionTransformer(**config)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    result = fit_model(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        nn.CrossEntropyLoss(),
        device,
        epochs,
        "vit",
        output_dir / "checkpoints" / "vit_best.pt",
        scheduler=scheduler,
        early_stopping_patience=8 if not args.smoke_test else 2,
        history_csv_path=output_dir / "metrics" / "vit_history.csv",
        history_json_path=output_dir / "metrics" / "vit_history.json",
        extra_checkpoint={"model_type": "vit", "model_config": config},
    )
    plot_training_curves(result["history"], "vit", output_dir)
    return model, result, loaders


def train_hybrid(args: argparse.Namespace, data_root: Path, output_dir: Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    loaders = get_cifar10_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        model_family="standard",
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        smoke_test=args.smoke_test,
    )
    epochs = 1 if args.smoke_test else args.epochs_hybrid
    model = HybridCNNMLP(**DEFAULT_HYBRID_CONFIG)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    result = fit_model(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        nn.CrossEntropyLoss(),
        device,
        epochs,
        "hybrid",
        output_dir / "checkpoints" / "hybrid_best.pt",
        scheduler=scheduler,
        early_stopping_patience=8 if not args.smoke_test else 2,
        history_csv_path=output_dir / "metrics" / "hybrid_history.csv",
        history_json_path=output_dir / "metrics" / "hybrid_history.json",
        extra_checkpoint={"model_type": "hybrid", "model_config": DEFAULT_HYBRID_CONFIG},
    )
    plot_training_curves(result["history"], "hybrid", output_dir)
    return model, result, loaders


def _combine_resnet_results(head_result: dict, finetune_result: dict | None) -> dict:
    if finetune_result is None:
        return head_result

    combined_history = [row.copy() for row in head_result["history"]]
    offset = len(combined_history)
    for row in finetune_result["history"]:
        adjusted = row.copy()
        adjusted["epoch"] = offset + adjusted["epoch"]
        combined_history.append(adjusted)

    total_time = head_result["total_training_time"] + finetune_result["total_training_time"]
    result = finetune_result.copy()
    result["history"] = combined_history
    result["total_training_time"] = total_time
    result["average_epoch_time"] = sum(row["epoch_time"] for row in combined_history) / max(len(combined_history), 1)
    result["peak_gpu_memory_mb"] = max(
        head_result["peak_gpu_memory_mb"],
        finetune_result["peak_gpu_memory_mb"],
    )
    return result


def train_resnet(args: argparse.Namespace, data_root: Path, output_dir: Path, device: torch.device) -> tuple[nn.Module, dict, dict]:
    loaders = get_cifar10_dataloaders(
        data_root=data_root,
        batch_size=args.resnet_batch_size,
        model_family="resnet",
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        smoke_test=args.smoke_test,
    )
    epochs = 1 if args.smoke_test else args.epochs_resnet
    model = create_resnet18_transfer(num_classes=10, pretrained=True, freeze_backbone=True)
    optimizer = Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    checkpoint_path = output_dir / "checkpoints" / "resnet_best.pt"

    head_result = fit_model(
        model,
        loaders["train"],
        loaders["val"],
        optimizer,
        nn.CrossEntropyLoss(),
        device,
        epochs,
        "resnet_head",
        checkpoint_path,
        scheduler=scheduler,
        early_stopping_patience=5 if not args.smoke_test else 2,
        history_csv_path=output_dir / "metrics" / "resnet_head_history.csv",
        history_json_path=output_dir / "metrics" / "resnet_head_history.json",
        extra_checkpoint={
            "model_type": "resnet",
            "model_config": {"num_classes": 10, "pretrained": False, "freeze_backbone": False},
            "normalization_mean": IMAGENET_MEAN,
            "normalization_std": IMAGENET_STD,
        },
    )

    finetune_result = None
    finetune_epochs = 1 if args.smoke_test else args.epochs_resnet_finetune
    if not args.skip_resnet_finetune and finetune_epochs > 0:
        unfreeze_layer4(model)
        optimizer = SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-4,
            momentum=0.9,
            weight_decay=1e-4,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        finetune_result = fit_model(
            model,
            loaders["train"],
            loaders["val"],
            optimizer,
            nn.CrossEntropyLoss(),
            device,
            finetune_epochs,
            "resnet_layer4_finetune",
            checkpoint_path,
            scheduler=scheduler,
            early_stopping_patience=4 if not args.smoke_test else 2,
            history_csv_path=output_dir / "metrics" / "resnet_finetune_history.csv",
            history_json_path=output_dir / "metrics" / "resnet_finetune_history.json",
            extra_checkpoint={
                "model_type": "resnet",
                "model_config": {"num_classes": 10, "pretrained": False, "freeze_backbone": False},
                "normalization_mean": IMAGENET_MEAN,
                "normalization_std": IMAGENET_STD,
            },
            initial_best_val_loss=head_result["best_val_loss"],
            initial_best_val_accuracy=head_result["best_val_accuracy"],
        )

    result = _combine_resnet_results(head_result, finetune_result)
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["history"] = result["history"]
        checkpoint["best_val_loss"] = result["best_val_loss"]
        checkpoint["best_val_accuracy"] = result["best_val_accuracy"]
        torch.save(checkpoint, checkpoint_path)
    save_history_csv(result["history"], output_dir / "metrics" / "resnet_history.csv")
    save_history_json(result["history"], output_dir / "metrics" / "resnet_history.json")
    plot_training_curves(result["history"], "resnet", output_dir)
    return model, result, loaders


def main() -> None:
    args = parse_args()
    set_seed(RANDOM_SEED)
    data_root = resolve_data_root(args.data_root)
    output_dir = resolve_output_dir(args.output_dir)
    ensure_output_dirs(output_dir)
    device = get_device()

    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"Device: {device}")

    preview_loaders = get_cifar10_dataloaders(
        data_root=data_root,
        batch_size=args.batch_size,
        model_family="standard",
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        smoke_test=args.smoke_test,
    )
    class_names = preview_loaders["class_names"]
    save_cifar10_samples(data_root, output_dir, class_names)
    save_augmentation_visualization(data_root, output_dir, class_names)

    vit_config = DEFAULT_VIT_CONFIG.copy()
    if args.tune_vit_fast:
        vit_config = tune_vit_fast(args, data_root, output_dir, device)

    summary_rows: list[dict] = []

    vit_model, vit_result, vit_loaders = train_vit(args, data_root, output_dir, device, vit_config)
    summary_rows.append(
        _evaluate_and_save(
            "vit",
            "ViT from scratch",
            vit_model,
            vit_result,
            vit_loaders["test"],
            class_names,
            output_dir,
            device,
            "standard",
        )
    )

    hybrid_model, hybrid_result, hybrid_loaders = train_hybrid(args, data_root, output_dir, device)
    summary_rows.append(
        _evaluate_and_save(
            "hybrid",
            "Hybrid CNN + MLP",
            hybrid_model,
            hybrid_result,
            hybrid_loaders["test"],
            class_names,
            output_dir,
            device,
            "standard",
        )
    )

    resnet_completed = False
    try:
        resnet_model, resnet_result, resnet_loaders = train_resnet(args, data_root, output_dir, device)
        summary_rows.append(
            _evaluate_and_save(
                "resnet",
                "Pretrained ResNet18 transfer learning",
                resnet_model,
                resnet_result,
                resnet_loaders["test"],
                class_names,
                output_dir,
                device,
                "resnet",
            )
        )
        resnet_completed = True
    except RuntimeError as exc:
        print("\nResNet training could not start:")
        print(exc)
        print("The ViT and Hybrid outputs were still saved. Cache the ImageNet weights once, then rerun.")

    save_metrics_summary(summary_rows, output_dir)
    save_comparison_markdown(summary_rows, output_dir)

    if args.smoke_test and resnet_completed:
        print("\nSmoke test completed successfully.")
    elif args.smoke_test:
        print("\nSmoke test completed for ViT and Hybrid; ResNet needs cached ImageNet weights to run.")
    else:
        print("\nTraining run completed.")
    print(f"Metrics: {output_dir / 'metrics'}")
    print(f"Checkpoints: {output_dir / 'checkpoints'}")
    print(f"Plots: {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
