"""Local deployment-style demo for saved CIFAR-10 classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision import datasets

from src.config import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, DEFAULT_HYBRID_CONFIG, DEFAULT_VIT_CONFIG, IMAGENET_MEAN, IMAGENET_STD, RANDOM_SEED, ensure_output_dirs, resolve_data_root, resolve_output_dir
from src.data import get_transforms, validate_cifar10_root
from src.models import HybridCNNMLP, VisionTransformer
from src.models.resnet_transfer import create_resnet18_transfer
from src.visualization import save_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local inference with a saved CIFAR-10 checkpoint.")
    parser.add_argument("--model", choices=["vit", "hybrid", "resnet"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _build_model(model_key: str, checkpoint: dict) -> torch.nn.Module:
    if model_key == "vit":
        config = DEFAULT_VIT_CONFIG.copy()
        config.update(checkpoint.get("model_config", {}))
        return VisionTransformer(**config)
    if model_key == "hybrid":
        config = DEFAULT_HYBRID_CONFIG.copy()
        config.update(checkpoint.get("model_config", {}))
        return HybridCNNMLP(**config)
    config = checkpoint.get("model_config", {"num_classes": 10})
    return create_resnet18_transfer(
        num_classes=config.get("num_classes", 10),
        pretrained=False,
        freeze_backbone=False,
    )


def main() -> None:
    args = parse_args()
    data_root = validate_cifar10_root(resolve_data_root(args.data_root))
    output_dir = resolve_output_dir(args.output_dir)
    ensure_output_dirs(output_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(__file__).resolve().parent / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _build_model(args.model, checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    model_family = "resnet" if args.model == "resnet" else "standard"
    _, eval_transform = get_transforms(model_family)
    dataset = datasets.CIFAR10(root=data_root, train=False, transform=eval_transform, download=False)

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    indices = torch.randperm(len(dataset), generator=generator).tolist()[: args.num_images]
    images = []
    labels = []
    for index in indices:
        image, label = dataset[index]
        images.append(image)
        labels.append(label)

    image_batch = torch.stack(images).to(device)
    labels_tensor = torch.tensor(labels)

    with torch.no_grad():
        logits = model(image_batch)
        probabilities = torch.softmax(logits, dim=1).cpu()
        predictions = probabilities.argmax(dim=1)

    print("\nLocal demo predictions")
    for row, (predicted, true_label) in enumerate(zip(predictions.tolist(), labels_tensor.tolist()), start=1):
        confidence = probabilities[row - 1, predicted].item()
        print(
            f"{row:02d}. predicted={CIFAR10_CLASSES[predicted]:>10s} "
            f"true={CIFAR10_CLASSES[true_label]:>10s} confidence={confidence:.3f}"
        )

    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if args.model == "resnet" else (CIFAR10_MEAN, CIFAR10_STD)
    output_path = output_dir / "report_assets" / "local_demo_predictions.png"
    save_prediction_grid(
        image_batch.cpu(),
        labels_tensor.numpy(),
        predictions.numpy(),
        probabilities.numpy(),
        CIFAR10_CLASSES,
        output_path,
        f"Local demo: {args.model}",
        mean=mean,
        std=std,
        max_images=args.num_images,
    )
    print(f"Saved prediction grid: {output_path}")


if __name__ == "__main__":
    main()

