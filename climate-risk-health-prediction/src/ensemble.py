import os
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, roc_auc_score


def align_probabilities(probs: np.ndarray, t_cut: float) -> np.ndarray:
    """
    Monotonically maps raw probabilities such that:
    - [0, t_cut] -> [0, 0.5]
    - [t_cut, 1] -> [0.5, 1.0]
    Preserves 100% of ROC-AUC ranking while ensuring TargetF1 = (TargetRAUC >= 0.5)
    operates at the optimal decision threshold t_cut.
    """
    t_cut = max(0.01, min(0.99, t_cut))
    aligned = np.where(
        probs <= t_cut,
        0.5 * (probs / t_cut),
        0.5 + 0.5 * ((probs - t_cut) / (1.0 - t_cut))
    )
    return np.clip(aligned, 0.0001, 0.9999)


def blend_predictions(oof_list, test_list, y_true, test_ids, model_names=None):
    """
    Blend multiple model predictions via optimized weights and calibrate threshold.
    """
    n_models = len(oof_list)
    oof_matrix = np.column_stack(oof_list)
    test_matrix = np.column_stack(test_list)

    if model_names is None:
        model_names = [f"Model_{i+1}" for i in range(n_models)]

    print(f"\nOptimizing weights across {n_models} models: {model_names}")

    def objective(w):
        w = np.maximum(0, w)
        w = w / (np.sum(w) + 1e-8)
        blend = oof_matrix @ w
        # Find best threshold on this blend
        best_f = 0
        for t in np.linspace(0.30, 0.60, 31):
            f = f1_score(y_true, (blend >= t).astype(int))
            if f > best_f:
                best_f = f
        auc = roc_auc_score(y_true, blend)
        return -(0.60 * best_f + 0.40 * auc)

    init_weights = np.ones(n_models) / n_models
    res = minimize(objective, init_weights, method="Nelder-Mead", options={"maxiter": 600})
    weights = np.maximum(0, res.x)
    weights = weights / np.sum(weights)

    print("\nOptimal Model Weights:")
    for name, w in zip(model_names, weights):
        print(f"  {name:<15}: {w:.4f}")

    # Compute raw blended OOF and Test
    raw_blended_oof = oof_matrix @ weights
    raw_blended_test = test_matrix @ weights

    # Find optimal threshold on blended OOF
    raw_auc = roc_auc_score(y_true, raw_blended_oof)
    raw_f1 = f1_score(y_true, (raw_blended_oof >= 0.5).astype(int))

    best_t = 0.5
    best_f1 = raw_f1
    for t in np.linspace(0.30, 0.60, 61):
        cur_f1 = f1_score(y_true, (raw_blended_oof >= t).astype(int))
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_t = t

    # Monotonic probability alignment
    aligned_blended_oof = align_probabilities(raw_blended_oof, best_t)
    aligned_blended_test = align_probabilities(raw_blended_test, best_t)

    final_auc = roc_auc_score(y_true, aligned_blended_oof)
    final_f1 = f1_score(y_true, (aligned_blended_oof >= 0.5).astype(int))
    final_score = 0.60 * final_f1 + 0.40 * final_auc

    print("------------------------------------------")
    print(f"RAW BLEND (thresh=0.50)  | F1: {raw_f1:.5f} | AUC: {raw_auc:.5f} | Score: {0.60*raw_f1 + 0.40*raw_auc:.5f}")
    print(f"ALIGNED BLEND (cut={best_t:.3f}) | F1: {final_f1:.5f} | AUC: {final_auc:.5f} | Score: {final_score:.5f}")
    print("------------------------------------------")

    sub_df = pd.DataFrame({
        "ID": test_ids,
        "TargetF1": (aligned_blended_test >= 0.5).astype(int),
        "TargetRAUC": aligned_blended_test
    })

    # Strict submission integrity verification
    assert len(sub_df) == 1030, f"Expected 1030 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["ID", "TargetF1", "TargetRAUC"], f"Invalid columns: {sub_df.columns}"
    assert sub_df["TargetRAUC"].isna().sum() == 0, "Submission contains NaNs in TargetRAUC!"
    assert sub_df["TargetF1"].isin([0, 1]).all(), "TargetF1 contains non-binary values!"
    assert (sub_df["TargetF1"] == (sub_df["TargetRAUC"] >= 0.5).astype(int)).all(), "Threshold violation!"

    metric = {
        "f1": final_f1,
        "roc_auc": final_auc,
        "final_score": final_score,
        "best_threshold": best_t,
        "weights": weights
    }

    return sub_df, metric
