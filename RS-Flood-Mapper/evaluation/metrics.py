# evaluation/metrics.py

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calculate_iou(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Intersection over Union (IoU) metric.

    Args:
        y_true (numpy.ndarray): The ground truth labels (flattened).
        y_pred (numpy.ndarray): The predicted labels (flattened).
        smooth (float): A small value to avoid division by zero.

    Returns:
        float: The IoU score.
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou


def print_evaluation_metrics(y_true, y_pred):
    """
    Calculates and prints a set of common evaluation metrics.

    Args:
        y_true (numpy.ndarray): The ground truth labels (2D image).
        y_pred (numpy.ndarray): The predicted labels (2D image).
    """
    # Flatten the arrays to compute metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    iou = calculate_iou(y_true_flat, y_pred_flat)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"IoU Score: {iou:.4f}")
    print("------------------------\n")