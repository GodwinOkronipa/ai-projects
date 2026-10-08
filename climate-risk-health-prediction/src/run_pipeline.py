import os
import numpy as np
import pandas as pd

from utils import set_seed, compute_zindi_metric
from features import prepare_datasets
from train import preprocess_data, train_cv
from ensemble import blend_predictions


def run_full_pipeline(data_dir: str = r"C:\Users\gokro\Desktop\climate-risk-health-prediction\data",
                      models_to_run=None):
    set_seed(42)
    if models_to_run is None:
        models_to_run = ["logistic", "xgb"]

    # Check for available optional models
    try:
        import lightgbm
        if "lgbm" not in models_to_run:
            models_to_run.append("lgbm")
    except ImportError:
        pass

    try:
        import catboost
        if "catboost" not in models_to_run:
            models_to_run.append("catboost")
    except ImportError:
        pass

    print(f"Starting End-to-End Pipeline with models: {models_to_run}")

    train_df, test_df = prepare_datasets(data_dir)
    y_true = train_df["is_climate_sensitive"].values
    test_ids = test_df["ID"].values

    oof_dict = {}
    test_dict = {}
    metric_dict = {}

    for m in models_to_run:
        try:
            oof, test_p, metric = train_cv(model_name=m, n_splits=5, seed=42, data_dir=data_dir)
            oof_dict[m] = oof
            test_dict[m] = test_p
            metric_dict[m] = metric
        except Exception as e:
            print(f"Error training {m}: {e}")

    # Summary table
    print("\n================ MODEL PERFORMANCE SUMMARY ================")
    print(f"{'Model':<15} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Final Score':<12}")
    print("-" * 55)
    for m, met in metric_dict.items():
        print(f"{m:<15} | {met['f1']:<10.5f} | {met['roc_auc']:<10.5f} | {met['final_score']:<12.5f}")
    print("=" * 55)

    # If more than 1 model ran, ensemble them!
    if len(oof_dict) > 1:
        print("\nCreating Ensemble Blend...")
        oof_list = list(oof_dict.values())
        test_list = list(test_dict.values())

        sub_blend, blend_met = blend_predictions(oof_list, test_list, y_true, test_ids, method="opt")
        blend_path = os.path.join(os.path.dirname(data_dir), "submissions", f"sub_ensemble_cv_{blend_met['final_score']:.4f}.csv")
        sub_blend.to_csv(blend_path, index=False)
        print(f"Ensemble submission saved to: {blend_path}")
        print(sub_blend.head(10))


if __name__ == "__main__":
    run_full_pipeline()
