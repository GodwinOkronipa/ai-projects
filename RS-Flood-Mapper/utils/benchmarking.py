"""
Model Benchmarking & Explainable AI (XAI) Engine for Remote Sensing Flood Delineation.

Calculates quantitative comparative metrics (mIoU, Dice, Precision, Recall, Latency)
and extracts spectral feature importances from Random Forest classifiers.
"""

import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latency_ms: float = 0.0
) -> Dict[str, float]:
    """
    Calculate full evaluation metrics between ground truth and predicted binary mask.
    """
    y_t = y_true.astype(bool).flatten()
    y_p = y_pred.astype(bool).flatten()

    tp = np.sum(y_t & y_p)
    fp = np.sum((~y_t) & y_p)
    fn = np.sum(y_t & (~y_p))
    tn = np.sum((~y_t) & (~y_p))

    intersection = float(tp)
    union = float(tp + fp + fn)

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (2.0 * intersection + fp + fn + 1e-7)
    precision = (intersection + 1e-7) / (intersection + fp + 1e-7)
    recall = (intersection + 1e-7) / (intersection + fn + 1e-7)
    accuracy = (tp + tn) / float(len(y_t))

    return {
        "mIoU": round(float(iou), 4),
        "Dice / F1": round(float(dice), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "Accuracy": round(float(accuracy), 4),
        "Latency (ms)": round(latency_ms, 2),
    }


def benchmark_all_models(
    features: np.ndarray,
    ground_truth: np.ndarray,
    rf_model,
    water_threshold: float = 0.15
) -> pd.DataFrame:
    """
    Run and compare Random Forest, NDWI Spectral Indexing, and Otsu Thresholding.
    """
    h, w = ground_truth.shape
    ndwi = features[:, :, 3] if features.shape[2] >= 4 else features[:, :, 0]
    num_pixels = h * w
    num_feats = features.shape[2]
    X_flat = features.reshape(num_pixels, num_feats)

    results = []

    # 1. Random Forest
    t0 = time.perf_counter()
    pred_rf_flat = rf_model.predict(X_flat)
    rf_latency = (time.perf_counter() - t0) * 1000.0
    pred_rf = pred_rf_flat.reshape((h, w)) > 0
    rf_metrics = compute_model_metrics(ground_truth, pred_rf, rf_latency)
    rf_metrics["Model"] = "Random Forest (ML)"
    results.append(rf_metrics)

    # 2. NDWI Index Cutoff
    t0 = time.perf_counter()
    pred_ndwi = ndwi >= water_threshold
    ndwi_latency = (time.perf_counter() - t0) * 1000.0
    ndwi_metrics = compute_model_metrics(ground_truth, pred_ndwi, ndwi_latency)
    ndwi_metrics["Model"] = "NDWI Spectral Cutoff"
    results.append(ndwi_metrics)

    # 3. Otsu Adaptive Thresholding
    t0 = time.perf_counter()
    norm_ndwi = ((ndwi + 1.0) / 2.0 * 255.0).astype(np.uint8)
    hist, _ = np.histogram(norm_ndwi, bins=256, range=(0, 256))
    total = float(norm_ndwi.size)
    current_max, threshold = 0.0, 128
    sum_total = np.dot(np.arange(256), hist)
    weight_bg, sum_bg = 0.0, 0.0
    
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        var_between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if var_between > current_max:
            current_max = var_between
            threshold = t
    otsu_cutoff = (threshold / 255.0) * 2.0 - 1.0
    pred_otsu = ndwi >= (otsu_cutoff * 0.9)
    otsu_latency = (time.perf_counter() - t0) * 1000.0
    otsu_metrics = compute_model_metrics(ground_truth, pred_otsu, otsu_latency)
    otsu_metrics["Model"] = "Adaptive Otsu Threshold"
    results.append(otsu_metrics)

    df = pd.DataFrame(results)
    # Put Model column first
    cols = ["Model", "mIoU", "Dice / F1", "Precision", "Recall", "Accuracy", "Latency (ms)"]
    return df[cols]


def extract_feature_importances(
    rf_model,
    feature_names: List[str] = ["Red (B04)", "Green (B03)", "Blue (B02)", "NDWI (Water Index)"]
) -> pd.DataFrame:
    """
    Extract and format Gini feature importances from trained Random Forest classifier.
    """
    if not hasattr(rf_model, "feature_importances_"):
        return pd.DataFrame()

    importances = rf_model.feature_importances_
    names = feature_names[:len(importances)]
    
    df = pd.DataFrame({
        "Spectral Band": names,
        "Importance (%)": np.round(importances * 100.0, 2),
        "Domain Role": [
            "Soil/Sediment Reflection",
            "Chlorophyll & Turbidity Peak",
            "Atmospheric Penetration",
            "Differential Water Absorption Ratio"
        ][:len(importances)]
    })
    return df.sort_values(by="Importance (%)", ascending=False).reset_index(drop=True)
