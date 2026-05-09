"""Hybrid CNN + MLP model for CIFAR-10 classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Small Conv-BatchNorm-ReLU block used by the hybrid baseline."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HybridCNNMLP(nn.Module):
    """Learn local image features with CNN blocks, then classify with an MLP.

    This baseline uses convolutional filters to learn local patch/image features
    and a fully connected MLP head for classification. It intentionally avoids
    self-attention and Transformer encoder blocks so it can be compared against
    the ViT from scratch.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.config = {"num_classes": num_classes, "dropout": dropout}
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 192),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 2 * 2, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(384, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

