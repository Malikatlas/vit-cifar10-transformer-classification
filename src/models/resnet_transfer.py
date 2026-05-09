"""ResNet18 transfer learning model for CIFAR-10."""

from __future__ import annotations

from torch import nn
from torchvision.models import resnet18


def create_resnet18_transfer(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Create a ResNet18 model with an ImageNet-pretrained backbone.

    The assignment requires ``torchvision.models.resnet18(weights="IMAGENET1K_V1")``.
    If the weights are not cached locally, torchvision may need internet access
    once. A clear RuntimeError is raised if the weights cannot be loaded.
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        if not pretrained:
            raise
        raise RuntimeError(
            "Could not load ResNet18 ImageNet weights. If they are not already "
            "cached, connect to the internet once and rerun training. Original "
            f"error: {exc}"
        ) from exc

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.config = {
        "num_classes": num_classes,
        "pretrained": pretrained,
        "freeze_backbone": freeze_backbone,
    }
    return model


def freeze_backbone_except_fc(model: nn.Module) -> nn.Module:
    """Freeze all layers except the final classifier."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.")
    return model


def unfreeze_layer4(model: nn.Module) -> nn.Module:
    """Unfreeze ResNet layer4 and the final classifier for fine-tuning."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("layer4.") or name.startswith("fc.")
    return model
