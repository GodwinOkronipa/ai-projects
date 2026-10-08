import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from utils import set_seed, compute_zindi_metric
from features import prepare_datasets
from models import get_model


def preprocess_data(train_df, test_df, target_col="is_climate_sensitive", id_col="ID"):
    """
    Separate features and target, apply categorical encoding.
    """
    y = train_df[target_col].values
    train_ids = train_df[id_col].values
    test_ids = test_df[id_col].values

    # Drop non-feature columns
    drop_cols = [id_col, target_col, "location"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    # Identify categorical and numeric columns
    cat_cols = [c for c in ["zone", "gender", "district"] if c in X_train.columns]
    num_cols = [c for c in X_train.columns if c not in cat_cols and c != "village"]

    # Simple frequency encoding for village if present
    if "village" in X_train.columns:
        freq = X_train["village"].value_counts(normalize=True)
        X_train["village_freq"] = X_train["village"].map(freq).fillna(0)
        X_test["village_freq"] = X_test["village"].map(freq).fillna(0)
        num_cols.append("village_freq")
        X_train = X_train.drop(columns=["village"])
        X_test = X_test.drop(columns=["village"])

    # One-hot encode categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    # Impute any residual NaNs with column medians
    col_medians = np.nanmedian(X_train_encoded, axis=0)
    col_medians = np.nan_to_num(col_medians, nan=0.0)

    for i in range(X_train_encoded.shape[1]):
        X_train_encoded[np.isnan(X_train_encoded[:, i]), i] = col_medians[i]
        X_test_encoded[np.isnan(X_test_encoded[:, i]), i] = col_medians[i]

    return X_train_encoded, y, X_test_encoded, test_ids


def train_cv(model_name: str = "lgbm", n_splits: int = 5, seed: int = 42, data_dir: str = "../data", **model_params):
    set_seed(seed)
    print(f"\n==========================================")
    print(f"Training {model_name.upper()} with {n_splits}-Fold Stratified CV (seed={seed})")
    print(f"==========================================")

    # 1. Load and engineer features
    train_df, test_df = prepare_datasets(data_dir)
    print(f"Train shape after feature engineering: {train_df.shape}")
    print(f"Test shape after feature engineering: {test_df.shape}")

    # 2. Preprocessing
    X, y, X_test, test_ids = preprocess_data(train_df, test_df)
    print(f"Feature matrix shape: {X.shape}, Target positive rate: {np.mean(y):.4f}")

    # 3. Stratified K-Fold CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_probs = np.zeros(len(X))
    test_probs = np.zeros(len(X_test))
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = get_model(model_name, random_state=seed + fold, **model_params)
        model.fit(X_tr, y_tr)

        # Predict probability for class 1
        val_prob = model.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = val_prob

        # Accumulate test prediction
        test_prob_fold = model.predict_proba(X_test)[:, 1]
        test_probs += test_prob_fold / n_splits

        # Fold metric
        m = compute_zindi_metric(y_val, val_prob)
        fold_metrics.append(m)
        print(f"Fold {fold}/{n_splits} | F1: {m['f1']:.5f} | AUC: {m['roc_auc']:.5f} | Final: {m['final_score']:.5f}")

    # 4. Overall OOF Metric
    overall_metric = compute_zindi_metric(y, oof_probs)
    print("------------------------------------------")
    print(f"OVERALL OOF F1-Score : {overall_metric['f1']:.5f}")
    print(f"OVERALL OOF ROC-AUC  : {overall_metric['roc_auc']:.5f}")
    print(f"OVERALL OOF SCORE    : {overall_metric['final_score']:.5f} (0.60*F1 + 0.40*AUC)")
    print("------------------------------------------")

    # 5. Create Submission
    sub_df = pd.DataFrame({
        "ID": test_ids,
        "TargetF1": (test_probs >= 0.5).astype(int),
        "TargetRAUC": test_probs
    })

    # Verification checks
    assert len(sub_df) == 1030, f"Expected 1030 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["ID", "TargetF1", "TargetRAUC"], f"Invalid columns: {sub_df.columns}"
    assert sub_df["TargetRAUC"].isna().sum() == 0, "Submission contains NaNs in TargetRAUC!"
    assert sub_df["TargetF1"].isin([0, 1]).all(), "TargetF1 contains non-binary values!"
    assert (sub_df["TargetF1"] == (sub_df["TargetRAUC"] >= 0.5).astype(int)).all(), "Threshold violation!"

    # Save submission
    sub_filename = f"sub_{model_name}_cv_{overall_metric['final_score']:.4f}.csv"
    sub_path = os.path.join(os.path.dirname(data_dir), "submissions", sub_filename)
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to: {sub_path}")
    print(sub_df.head(10))

    return oof_probs, test_probs, overall_metric


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lgbm", choices=["logistic", "lgbm", "xgb", "catboost"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=r"C:\Users\gokro\Desktop\climate-risk-health-prediction\data")
    args = parser.parse_args()

    train_cv(model_name=args.model, n_splits=args.folds, seed=args.seed, data_dir=args.data_dir)
