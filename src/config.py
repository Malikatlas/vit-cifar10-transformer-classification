"""Shared configuration values for the CIFAR-10 ViT assignment."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT
OUTPUT_DIR = PROJECT_ROOT / "outputs"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORT_ASSETS_DIR = OUTPUT_DIR / "report_assets"
REPORTS_DIR = OUTPUT_DIR / "reports"

RANDOM_SEED = 42
NUM_WORKERS = 2
NUM_CLASSES = 10

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_VIT_CONFIG = {
    "image_size": 32,
    "patch_size": 4,
    "num_classes": NUM_CLASSES,
    "embed_dim": 192,
    "depth": 6,
    "num_heads": 6,
    "mlp_dim": 384,
    "dropout": 0.1,
}

DEFAULT_HYBRID_CONFIG = {
    "num_classes": NUM_CLASSES,
    "dropout": 0.25,
}


def ensure_output_dirs(output_dir: Path = OUTPUT_DIR) -> None:
    """Create the output folder tree used by training, evaluation, and reports."""
    for relative in [
        "checkpoints",
        "plots",
        "metrics",
        "report_assets",
        "reports",
    ]:
        (Path(output_dir) / relative).mkdir(parents=True, exist_ok=True)


def resolve_data_root(data_root: str | Path | None = None) -> Path:
    """Return a normalized dataset root path."""
    return Path(data_root).expanduser().resolve() if data_root else DATA_ROOT.resolve()


def resolve_output_dir(output_dir: str | Path | None = None) -> Path:
    """Return a normalized output path."""
    return Path(output_dir).expanduser().resolve() if output_dir else OUTPUT_DIR.resolve()

