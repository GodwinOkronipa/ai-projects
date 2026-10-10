import os
import shutil
import numpy as np
import pandas as pd

from utils import set_seed
from features import prepare_datasets
from train import train_cv
from ensemble import blend_predictions


def run_full_pipeline(data_dir: str = r"C:\Users\gokro\Desktop\climate-risk-health-prediction\data",
                      models_to_run=None,
                      n_splits: int = 10,
                      seed: int = 42):
    set_seed(seed)
    if models_to_run is None:
        models_to_run = ["lgbm", "xgb", "histgb", "extratrees", "logistic"]

    print(f"============================================================")
    print(f"  STARTING LEADERBOARD-MAXXING END-TO-END PIPELINE (v2)")
    print(f"  Models: {models_to_run}")
    print(f"  Validation: {n_splits}-Fold Stratified Cross-Validation")
    print(f"============================================================\n")

    train_df, test_df = prepare_datasets(data_dir)
    y_true = train_df["is_climate_sensitive"].values
    test_ids = test_df["ID"].values

    oof_dict = {}
    test_dict = {}
    metric_dict = {}

    for m in models_to_run:
        try:
            oof, test_p, metric = train_cv(
                model_name=m,
                n_splits=n_splits,
                seed=seed,
                data_dir=data_dir
            )
            oof_dict[m] = oof
            test_dict[m] = test_p
            metric_dict[m] = metric
        except Exception as e:
            print(f"Error training {m}: {e}")

    # Summary table
    print("\n================ MODEL PERFORMANCE SUMMARY ================")
    print(f"{'Model':<15} | {'F1-Score':<10} | {'ROC-AUC':<10} | {'Final Score':<12} | {'Opt Cut':<8}")
    print("-" * 65)
    for m, met in metric_dict.items():
        print(f"{m:<15} | {met['f1']:<10.5f} | {met['roc_auc']:<10.5f} | {met['final_score']:<12.5f} | {met['best_threshold']:<8.3f}")
    print("=" * 65)

    # Master Ensemble Blend
    if len(oof_dict) > 1:
        print("\nCreating Master Ensemble Blend...")
        model_names = list(oof_dict.keys())
        oof_list = [oof_dict[m] for m in model_names]
        test_list = [test_dict[m] for m in model_names]

        sub_blend, blend_met = blend_predictions(
            oof_list, test_list, y_true, test_ids, model_names=model_names
        )

        sub_dir = os.path.join(os.path.dirname(data_dir), "submissions")
        os.makedirs(sub_dir, exist_ok=True)
        blend_filename = f"sub_v2_maxxx_ensemble_cv_{blend_met['final_score']:.4f}.csv"
        blend_path = os.path.join(sub_dir, blend_filename)
        sub_blend.to_csv(blend_path, index=False)
        print(f"\n>>> MASTER ENSEMBLE SUBMISSION SAVED: {blend_path}")
        print(f">>> CV FINAL SCORE: {blend_met['final_score']:.5f} (F1: {blend_met['f1']:.5f}, AUC: {blend_met['roc_auc']:.5f})")
        print("\nFirst 10 rows of final submission:")
        print(sub_blend.head(10))

        # Copy to GitHub repo
        github_repo_sub_dir = r"C:\Users\gokro\Documents\GitHub\ai-projects\climate-risk-health-prediction\submissions"
        if os.path.exists(os.path.dirname(github_repo_sub_dir)):
            os.makedirs(github_repo_sub_dir, exist_ok=True)
            dst_path = os.path.join(github_repo_sub_dir, blend_filename)
            shutil.copyfile(blend_path, dst_path)
            print(f"Copied master submission to GitHub repo: {dst_path}")


if __name__ == "__main__":
    run_full_pipeline()
