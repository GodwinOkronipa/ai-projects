"""
Evaluation metrics calculation module for Flood Extent Semantic Segmentation.

Author: godmode-dev
License: MIT
"""

import numpy as np
import torch
from typing import Dict


def calculate_segmentation_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate Mean Intersection over Union (mIoU), Dice Coefficient, Precision, and Recall.
    """
    y_pred_binary = (y_pred >= threshold).float()
    y_true_binary = y_true.float()

    intersection = (y_pred_binary * y_true_binary).sum()
    union = y_pred_binary.sum() + y_true_binary.sum() - intersection
    total_pred = y_pred_binary.sum()
    total_true = y_true_binary.sum()

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (y_pred_binary.sum() + y_true_binary.sum() + 1e-7)
    precision = (intersection + 1e-7) / (total_pred + 1e-7)
    recall = (intersection + 1e-7) / (total_true + 1e-7)
    f1 = (2 * precision * recall) / (precision + recall + 1e-7)

    return {
        "mIoU": float(iou.item()),
        "dice_score": float(dice.item()),
        "precision": float(precision.item()),
        "recall": float(recall.item()),
        "f1_score": float(f1.item()),
    }
