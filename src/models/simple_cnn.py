"""Simple CNN architecture used by the TeaVision training scripts."""

from pathlib import Path

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Small convolutional classifier for resized RGB tea-leaf images."""

    def __init__(self, num_classes: int, image_size: int = 128, dropout: float = 0.0):
        super().__init__()
        if image_size < 8:
            raise ValueError("image_size must be at least 8 pixels.")

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        feature_size = image_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * feature_size * feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def load_simple_cnn_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[SimpleCNN, dict]:
    """Load a TeaVision SimpleCNN checkpoint and return an evaluation model."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required_keys = {"model_state_dict", "image_size", "class_names"}
    missing_keys = required_keys.difference(checkpoint)
    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise ValueError(f"Checkpoint is missing required keys: {missing}")

    model = SimpleCNN(
        num_classes=len(checkpoint["class_names"]),
        image_size=int(checkpoint["image_size"]),
        dropout=float(checkpoint.get("cnn_dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
