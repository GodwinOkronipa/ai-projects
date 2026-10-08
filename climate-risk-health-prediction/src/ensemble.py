import os
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils import compute_zindi_metric


def blend_predictions(oof_list, test_list, y_true, test_ids, method="opt"):
    """
    Blend multiple model predictions.
    methods:
    - 'equal': Equal weighting
    - 'opt': Optimize weights to maximize Zindi Final Score
    """
    n_models = len(oof_list)

    if method == "equal":
        weights = np.ones(n_models) / n_models
    else:
        # Optimization objective: minimize negative Zindi final score
        def objective(w):
            w = np.array(w)
            w = w / np.sum(w)
            blended_oof = np.zeros_like(oof_list[0])
            for i in range(n_models):
                blended_oof += w[i] * oof_list[i]
            score = compute_zindi_metric(y_true, blended_oof)["final_score"]
            return -score

        init_weights = np.ones(n_models) / n_models
        bounds = [(0.0, 1.0) for _ in range(n_models)]
        res = minimize(objective, init_weights, bounds=bounds, method="Nelder-Mead")
        weights = res.x / np.sum(res.x)

    print("\nModel weights:")
    for i, w in enumerate(weights):
        print(f"  Model {i+1}: {w:.4f}")

    # Compute final blended OOF
    blended_oof = np.zeros_like(oof_list[0])
    for i in range(n_models):
        blended_oof += weights[i] * oof_list[i]

    metric = compute_zindi_metric(y_true, blended_oof)
    print("------------------------------------------")
    print(f"BLENDED OOF F1-Score : {metric['f1']:.5f}")
    print(f"BLENDED OOF ROC-AUC  : {metric['roc_auc']:.5f}")
    print(f"BLENDED OOF SCORE    : {metric['final_score']:.5f}")
    print("------------------------------------------")

    # Blend test predictions
    blended_test = np.zeros_like(test_list[0])
    for i in range(n_models):
        blended_test += weights[i] * test_list[i]

    sub_df = pd.DataFrame({
        "ID": test_ids,
        "TargetF1": (blended_test >= 0.5).astype(int),
        "TargetRAUC": blended_test
    })

    return sub_df, metric
