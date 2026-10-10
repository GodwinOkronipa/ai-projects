import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from utils import set_seed, compute_zindi_metric
from features import prepare_datasets
from models import get_model


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


def preprocess_data(train_df, test_df, target_col="is_climate_sensitive", id_col="ID"):
    """
    Separate features and target.
    """
    y = train_df[target_col].values
    train_ids = train_df[id_col].values
    test_ids = test_df[id_col].values

    drop_cols = [id_col, target_col]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Fill any potential NaNs with column medians
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test = X_test.fillna(medians)

    return X_train, y, X_test, test_ids


def train_cv(model_name: str = "lgbm", n_splits: int = 10, seed: int = 42, data_dir: str = "../data", **model_params):
    set_seed(seed)
    print(f"\n==========================================")
    print(f"Training {model_name.upper()} with {n_splits}-Fold Stratified CV (seed={seed})")
    print(f"==========================================")

    # 1. Load and engineer curated features
    train_df, test_df = prepare_datasets(data_dir)
    print(f"Train shape after feature engineering: {train_df.shape}")
    print(f"Test shape after feature engineering: {test_df.shape}")

    # 2. Preprocessing
    X, y, X_test, test_ids = preprocess_data(train_df, test_df)
    print(f"Features ({X.shape[1]}): {X.columns.tolist()}")

    # 3. Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_probs = np.zeros(len(X))
    test_probs = np.zeros(len(X_test))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        model = get_model(model_name, random_state=seed + fold, **model_params)
        model.fit(X_tr, y_tr)

        # Predict probability for class 1
        val_prob = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_prob

        # Accumulate test prediction
        test_prob_fold = model.predict_proba(X_test)[:, 1]
        test_probs += test_prob_fold / n_splits

        # Unaligned fold metric for monitoring
        m = compute_zindi_metric(y_val, val_prob)
        fold_metrics.append(m)
        print(f"Fold {fold:2d}/{n_splits} | F1 (0.5): {m['f1']:.5f} | AUC: {m['roc_auc']:.5f}")

    # 4. Find optimal threshold on OOF probabilities
    raw_auc = roc_auc_score(y, oof_probs)
    raw_f1 = f1_score(y, (oof_probs >= 0.5).astype(int))

    best_t = 0.5
    best_f1 = raw_f1
    for t in np.linspace(0.30, 0.60, 61):
        cur_f1 = f1_score(y, (oof_probs >= t).astype(int))
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_t = t

    # Monotonic alignment
    oof_aligned = align_probabilities(oof_probs, best_t)
    test_aligned = align_probabilities(test_probs, best_t)

    aligned_auc = roc_auc_score(y, oof_aligned)
    aligned_f1 = f1_score(y, (oof_aligned >= 0.5).astype(int))
    final_score = 0.60 * aligned_f1 + 0.40 * aligned_auc

    print("------------------------------------------")
    print(f"RAW OOF (thresh=0.50)  | F1: {raw_f1:.5f} | AUC: {raw_auc:.5f} | Score: {0.60*raw_f1 + 0.40*raw_auc:.5f}")
    print(f"ALIGNED OOF (cut={best_t:.3f}) | F1: {aligned_f1:.5f} | AUC: {aligned_auc:.5f} | Score: {final_score:.5f}")
    print("------------------------------------------")

    # 5. Create Submission
    sub_df = pd.DataFrame({
        "ID": test_ids,
        "TargetF1": (test_aligned >= 0.5).astype(int),
        "TargetRAUC": test_aligned
    })

    # Verification checks
    assert len(sub_df) == 1030, f"Expected 1030 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["ID", "TargetF1", "TargetRAUC"], f"Invalid columns: {sub_df.columns}"
    assert sub_df["TargetRAUC"].isna().sum() == 0, "Submission contains NaNs in TargetRAUC!"
    assert sub_df["TargetF1"].isin([0, 1]).all(), "TargetF1 contains non-binary values!"
    assert (sub_df["TargetF1"] == (sub_df["TargetRAUC"] >= 0.5).astype(int)).all(), "Threshold violation!"

    # Save submission
    sub_filename = f"sub_{model_name}_cv_{final_score:.4f}.csv"
    sub_path = os.path.join(os.path.dirname(data_dir), "submissions", sub_filename)
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to: {sub_path}")
    print(sub_df.head(5))

    metric_result = {
        "f1": aligned_f1,
        "roc_auc": aligned_auc,
        "final_score": final_score,
        "best_threshold": best_t
    }

    return oof_probs, test_probs, metric_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lgbm", choices=["logistic", "lgbm", "xgb", "histgb", "extratrees"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\gokro\Desktop\climate-risk-health-prediction\data")
    args = parser.parse_args()

    train_cv(model_name=args.model, n_splits=args.folds, seed=args.seed, data_dir=args.data_dir)
