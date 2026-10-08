import random
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def compute_zindi_metric(y_true, y_prob, threshold: float = 0.5):
    """
    Compute official challenge metric:
    Final Score = 0.60 * F1 + 0.40 * ROC-AUC

    Note: Rules strictly mandate default threshold of 0.5 for F1.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_prob))
    final_score = 0.60 * f1 + 0.40 * auc

    return {
        "final_score": final_score,
        "f1": f1,
        "roc_auc": auc,
    }
