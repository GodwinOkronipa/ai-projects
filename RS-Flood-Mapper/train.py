"""
PyTorch Deep Learning Training Pipeline for Satellite Remote Sensing Flood Mapping.

Author: godmode-dev
License: MIT
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from models.unet import UNet
from preprocessing.data_loader import get_dataloaders
from evaluation.evaluate import calculate_segmentation_metrics


class DiceBCELoss(nn.Module):
    """Combined Binary Cross Entropy and Dice Loss for semantic segmentation."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        intersection = (pred * target).sum()
        dice_loss = 1.0 - (2.0 * intersection + 1e-7) / (pred.sum() + target.sum() + 1e-7)
        return bce_loss + dice_loss


def train_model(
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_path: str = "models/unet_flood_model.pth"
) -> UNet:
    """Train PyTorch UNet model on multi-spectral satellite imagery."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=batch_size)

    model = UNet(in_channels=4, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n--- Starting PyTorch U-Net Training Loop ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_metrics: Dict[str, float] = {"mIoU": 0.0, "dice_score": 0.0}
        val_samples = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                batch_metrics = calculate_segmentation_metrics(masks, outputs)
                for k in val_metrics:
                    val_metrics[k] += batch_metrics[k] * images.size(0)
                val_samples += images.size(0)

        for k in val_metrics:
            val_metrics[k] /= val_samples

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | "
            f"Val mIoU: {val_metrics['mIoU']:.4f} | Val Dice: {val_metrics['dice_score']:.4f}"
        )

    torch.save(model.state_dict(), save_path)
    print(f"\nModel checkpoint saved successfully to {save_path}")
    return model


if __name__ == "__main__":
    train_model(epochs=3)