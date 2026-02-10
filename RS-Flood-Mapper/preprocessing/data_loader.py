"""
Data Loader and Synthetic Remote Sensing Image Generator for PyTorch Flood Segmentation.

Author: godmode-dev
License: MIT
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def generate_synthetic_flood_dataset(
    num_samples: int = 50,
    height: int = 128,
    width: int = 128,
    num_channels: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic multi-spectral satellite imagery and corresponding binary flood segmentation masks.
    Channels: [Sentinel-1 VV, Sentinel-1 VH, Sentinel-2 NDWI, Sentinel-2 NIR]
    """
    np.random.seed(42)
    images = np.random.randn(num_samples, num_channels, height, width).astype(np.float32)
    masks = np.zeros((num_samples, 1, height, width), dtype=np.float32)

    for i in range(num_samples):
        # Generate synthetic water body blob
        cx, cy = np.random.randint(30, width - 30), np.random.randint(30, height - 30)
        radius = np.random.randint(15, 35)
        
        y_grid, x_grid = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
        water_region = dist_from_center <= radius
        
        # High NDWI and low SAR backscatter in water region
        images[i, 2, water_region] += 2.5  # High NDWI
        images[i, 0, water_region] -= 2.0  # Low S1 VV
        images[i, 1, water_region] -= 2.0  # Low S1 VH
        
        masks[i, 0, water_region] = 1.0

    return images, masks


class FloodSegmentationDataset(Dataset):
    """
    PyTorch Dataset wrapper for multi-spectral remote sensing satellite tensors.
    """

    def __init__(self, images: np.ndarray, masks: np.ndarray) -> None:
        self.images = torch.from_numpy(images).float()
        self.masks = torch.from_numpy(masks).float()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.masks[idx]


def get_dataloaders(
    batch_size: int = 8,
    train_split: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch Train and Validation DataLoaders.
    """
    images, masks = generate_synthetic_flood_dataset(num_samples=60, height=128, width=128)
    split_idx = int(len(images) * train_split)

    train_ds = FloodSegmentationDataset(images[:split_idx], masks[:split_idx])
    val_ds = FloodSegmentationDataset(images[split_idx:], masks[split_idx:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader