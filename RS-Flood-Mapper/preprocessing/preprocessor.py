"""
Preprocessing and Index Calculation Utilities for Sentinel-1 SAR and Sentinel-2 Optical Remote Sensing Data.

Author: godmode-dev
License: MIT
"""

import numpy as np
from typing import Tuple


def apply_speckle_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Lee / Median speckle noise reduction filter to Sentinel-1 SAR backscatter imagery.
    """
    if img.ndim == 2:
        img_expanded = img[np.newaxis, :, :]
    else:
        img_expanded = img

    filtered = np.copy(img_expanded)
    pad = kernel_size // 2
    
    for c in range(img_expanded.shape[0]):
        padded = np.pad(img_expanded[c], pad_width=pad, mode="reflect")
        for i in range(img_expanded.shape[1]):
            for j in range(img_expanded.shape[2]):
                patch = padded[i : i + kernel_size, j : j + kernel_size]
                filtered[c, i, j] = np.median(patch)

    return filtered.squeeze()


def calculate_ndwi(green_band: np.ndarray, nir_band: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Difference Water Index (NDWI) from Sentinel-2 Green (B03) and NIR (B08) bands.
    NDWI = (Green - NIR) / (Green + NIR)
    """
    green = green_band.astype(np.float32)
    nir = nir_band.astype(np.float32)
    
    denom = green + nir
    denom[denom == 0] = 1e-10
    
    ndwi = (green - nir) / denom
    return np.clip(ndwi, -1.0, 1.0)


def normalize_image(img: np.ndarray, percentile_min: float = 2.0, percentile_max: float = 98.0) -> np.ndarray:
    """
    Min-Max robust scaling normalization using percentiles to eliminate extreme outliers.
    """
    img_float = img.astype(np.float32)
    p_min = np.percentile(img_float, percentile_min)
    p_max = np.percentile(img_float, percentile_max)

    if p_max == p_min:
        return np.zeros_like(img_float)

    normalized = (img_float - p_min) / (p_max - p_min)
    return np.clip(normalized, 0.0, 1.0)
