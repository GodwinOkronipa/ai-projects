# 🌍 Climate Risk & Health Prediction AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-LightGBM%20%7C%20XGBoost%20%7C%20HistGB%20%7C%20ExtraTrees%20%7C%20Scikit--Learn-orange.svg)]()
[![Competition](https://img.shields.io/badge/Competition-Zindi%20Challenge-green.svg)](https://zindi.world/competitions/climate-risk-health-prediction-challenge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

High-performance machine learning pipeline developed for the **Zindi Climate Risk and Health Prediction Challenge**. The model predicts whether recorded mortality cases fall into climate-sensitive categories by fusing health, demographic, temporal, geographic, and multi-source satellite indicators (CHIRPS precipitation, ERA5-Land reanalysis, MODIS NDVI, SRTM elevation).

---

## 🌟 Executive Summary & Key Milestones

- **Challenge Objective**: Detect climate-sensitive mortality patterns to enable timely health interventions in vulnerable populations in Uganda.
- **Official Metric**:
  $$\text{Final Score} = 0.60 \times \text{F1} + 0.40 \times \text{ROC-AUC}$$
- **Submission Requirements**:
  - `ID`: 1,030 test cases
  - `TargetF1`: Binary predictions $\{0, 1\}$
  - `TargetRAUC`: Continuous well-calibrated odds in $[0, 1]$
  - Strictly adhering to $\text{TargetF1} = (\text{TargetRAUC} \ge 0.5).\text{astype}(\text{int})$
- **Performance Evolution**:
  - Starter Baseline: `0.7631` (Public LB)
  - Initial Pipeline v1: `0.7988` (CV) $\to$ `0.8223` (Public LB, Rank 649)
  - **Upgraded Pipeline v2 (Leaderboard-Maxxing)**:
    - **Single Best Model (LightGBM)**: **0.82294** CV (F1: 0.82174, ROC-AUC: 0.82475)
    - **Master 5-Model Ensemble Blend**: **0.82261** CV (F1: 0.82076, **ROC-AUC: 0.82539**)

---

## 🔬 Breakthrough Discoveries & Feature Engineering

Through exhaustive empirical forward-selection and ablation studies, we identified the true predictive drivers while removing noisy features that caused overfitting:

1. **The Childhood Vulnerability Asymmetry**:
   - Over **55.8%** of all recorded mortality cases are under age 5.
   - For children aged 1–3, the climate-sensitive mortality rate is **93.6% – 95.9%** (malaria, severe diarrhea, acute respiratory infections).
   - In contrast, elderly individuals (age $\ge 60$) have a climate-sensitive mortality rate of only **29.2%** (predominantly dying from non-communicable diseases: stroke, cardiovascular, neoplasms).
   - *Key architectural fix*: Separated infant/child vulnerability from senior vulnerability, replacing contradictory combined age groups.

2. **The 15-Year Epidemiological Shift (`year_x_under5`)**:
   - In 2007–2011, under-5 mortality was **95.2%** climate-sensitive.
   - By 2017–2022, national health programs reduced this proportion to **61.4%**.
   - Modeling the non-linear interaction between `year` and childhood status (`year_x_infant`, `year_x_under5`) captured this secular trend.

3. **Curated Climate & Environmental Interactions**:
   - `under5_x_rain_days_30d`: High rain frequency creates mosquito breeding habitats, multiplying infant malaria risk.
   - `under5_x_ndvi_30d`: Vegetation greenness tracking vector habitats.
   - `senior_x_temp_diff_30d`: Temperature anomaly stress specifically impacting cardiovascular risk in older adults.
   - **Pruned Features**: Constant columns (e.g. `hot_days_30d`), high-noise daily rainfall metrics (`max_daily_rain_30d`), and micro-village frequency encoding that evaluated to zero on the test set were removed.

4. **Monotonic Probability Alignment**:
   - In raw probability space, the F1-maximizing decision threshold is approximately $0.38 - 0.42$ due to class distribution ($65.07\%$ positive rate).
   - We designed a piecewise linear monotonic probability mapping $f(p): [0, 1] \to [0, 1]$ that centers the optimal threshold exactly at $0.50$.
   - **Mathematical Guarantee**: Because $f$ is strictly monotonic, the relative ranking of all samples is $100\%$ preserved (`roc_auc_score` delta = $0.00000000$), while `TargetF1` achieves optimal F1 performance at threshold $0.50$.

---

## 📊 10-Fold Stratified Cross-Validation Benchmarks

Validation results across 10 folds on the complete training set (3,146 samples):

| Model Architecture | Out-of-Fold F1 | Out-of-Fold ROC-AUC | **Official Final Score** | Optimal Cut |
| :--- | :---: | :---: | :---: | :---: |
| *Baseline v1 Pipeline* | *0.7915* | *0.8098* | *0.7988* | *0.500* |
| **Calibrated Logistic Regression** | 0.81779 | 0.81169 | **0.81535** | 0.370 |
| **ExtraTrees Classifier (Depth 6)** | 0.82010 | 0.82383 | **0.82159** | 0.425 |
| **HistGradientBoosting Classifier** | 0.82038 | 0.82394 | **0.82180** | 0.375 |
| **XGBoost Classifier (Depth 4)** | 0.82038 | 0.82446 | **0.82201** | 0.370 |
| **LightGBM Classifier (Depth 4)** | **0.82174** | 0.82475 | **0.82294** | 0.360 |
| **Master Ensemble Blend (5 Models)** | 0.82076 | **0.82539** | **0.82261** | 0.410 |

### Optimal Ensemble Weights
- **XGBoost**: 21.63%
- **ExtraTrees**: 20.21%
- **LightGBM**: 19.59%
- **HistGradientBoosting**: 19.43%
- **Logistic Regression**: 19.14%

---

## 📂 Repository Structure

```text
climate-risk-health-prediction/
├── data/
│   ├── Train.csv                     # 3,146 training records
│   ├── Test.csv                      # 1,030 test records
│   ├── climate_features.csv          # 4,176 climate records (CHIRPS, ERA5, MODIS, SRTM)
│   ├── SampleSubmission.csv          # Submission format template
│   └── data_dictionary.csv           # Dictionary for main attributes
├── src/
│   ├── utils.py                      # Multi-metric scorer & reproducibility seed utilities
│   ├── features.py                   # High-signal curated feature extraction pipeline
│   ├── models.py                     # LightGBM, XGBoost, HistGB, ExtraTrees, Logistic model factory
│   ├── train.py                      # 10-Fold Stratified CV engine with monotonic probability alignment
│   ├── ensemble.py                   # Nelder-Mead simplex ensemble weight optimizer
│   └── run_pipeline.py               # End-to-end multi-model orchestration script
├── submissions/
│   ├── sub_v2_maxxx_ensemble_cv_0.8226.csv   # 🥇 Master 5-Model Ensemble (CV: 0.8226, AUC: 0.8254)
│   ├── sub_lgbm_cv_0.8229.csv                # 🥈 Best Single Model: LightGBM (CV: 0.8229)
│   ├── sub_xgb_cv_0.8220.csv                 # 🥉 XGBoost (CV: 0.8220)
│   └── sub_ensemble_cv_0.7988.csv            # Baseline v1 submission
├── Climate_Risk_AI_Beginners_Guide_Blog.docx # Complete beginner-friendly deep-dive document
└── README.md                                 # Technical documentation
```

---

## 🚀 Running the Pipeline

```bash
# Execute the full 10-fold cross-validation pipeline & generate submissions
python src/run_pipeline.py
```
