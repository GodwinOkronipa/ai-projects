# 🌍 Climate Risk & Health Prediction AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-LightGBM%20%7C%20XGBoost%20%7C%20Scikit--Learn%20%7C%20Pandas-orange.svg)]()
[![Competition](https://img.shields.io/badge/Competition-Zindi%20Challenge-green.svg)](https://zindi.world/competitions/climate-risk-health-prediction-challenge)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

Supervised machine learning pipeline developed for the **Zindi Climate Risk and Health Prediction Challenge**. The model predicts whether recorded mortality cases fall into climate-sensitive categories by fusing health, demographic, geographic, and multi-source climate/satellite indicators.

---

## 🌟 Executive Summary

- **Challenge Objective**: Identify complex relationships between climate extremes and mortality vulnerability to support data-driven public health interventions in low-resource settings.
- **Evaluation Metric**: Official Zindi multi-metric combining precision-recall balance and ranking capability:
  $$\text{Final Score} = 0.60 \times \text{F1} + 0.40 \times \text{ROC-AUC}$$
- **Key Constraint**: Custom thresholding is strictly forbidden; `TargetF1` must be computed with the default threshold of `0.5`:
  $$\text{TargetF1} = (\text{TargetRAUC} \ge 0.5).\text{astype}(\text{int})$$
- **Performance**:
  - Baseline Starter Model: ~0.7938
  - **Single Best Model (XGBoost)**: **0.79871** (F1: 0.7958, ROC-AUC: 0.8030)
  - **Optimal Ensemble Blend**: **0.79880** (F1: 0.7915, **ROC-AUC: 0.80977**)

---

## 🔬 Domain Feature Engineering

The raw datasets (`Train.csv`, `Test.csv`, `climate_features.csv`) are transformed from 28 columns into **78 predictive features**:

1. **Temporal & Seasonal Dynamics**:
   - Date decomposition: `month`, `day_of_year`, `quarter`, `week_of_year`.
   - Harmonic cyclic representations: $\sin(2\pi \cdot \text{day\_of\_year} / 365.25)$, $\cos(2\pi \cdot \text{day\_of\_year} / 365.25)$, $\sin(2\pi \cdot \text{month} / 12)$, $\cos(2\pi \cdot \text{month} / 12)$.
   - East African (Uganda) bimodal rainfall indicators: long rainy season (March–May), short rainy season (Sept–Nov), and dry seasons.
2. **Meteorological Anomalies & Extreme Weather**:
   - Diurnal temperature range: $\text{max\_temp} - \text{min\_temp}$.
   - Temperature deviations vs short-term (7d), medium-term (30d), and seasonal (90d) baselines from ERA5-Land.
   - Temperature range anomaly: $\text{temp\_range\_daily} - \text{temp\_range\_mean\_30d}$.
   - Precipitation surplus/deficit vs 30-day average: $\text{precipitation} - (\text{rain\_sum\_30d} / 30)$.
   - Rainfall intensity ratios: $\text{rain\_sum\_7d} / \text{rain\_sum\_30d}$ and $\text{rain\_sum\_30d} / \text{rain\_sum\_90d}$.
   - Vegetation stress & NDVI dynamics: $\text{ndvi\_30d} - \text{ndvi\_90d}$.
3. **Demographic Vulnerability Interactions**:
   - Age strata flags: infants/under-5 ($\text{age} \le 5$), elderly ($\text{age} \ge 60$), overall vulnerable age.
   - Cross-interactions: vulnerable age $\times$ extreme heat days (`hot_days_30d`), vulnerable age $\times$ 30-day cumulative rainfall.
4. **Topography & Spatial Indices**:
   - Terrain ruggedness: $\text{elevation} \times \text{slope}$.
   - Administrative district and village extractions with frequency encoding.

---

## 📊 Cross-Validation Performance

5-Fold Stratified Cross-Validation on the training set (3,146 samples, 65.07% positive class), evaluated strictly at threshold 0.5:

| Model | Out-of-Fold F1 | Out-of-Fold ROC-AUC | **Official Final Score** |
| :--- | :---: | :---: | :---: |
| *Starter Notebook Baseline* | *~0.8019* | *~0.7817* | *0.79386* |
| **Calibrated Logistic Regression** | 0.76385 | 0.80394 | **0.77988** |
| **LightGBM Classifier** | 0.79438 | 0.80292 | **0.79780** |
| **XGBoost Classifier** | **0.79583** | 0.80303 | **0.79871** |
| **Ensemble Blend (Logistic + XGB + LGBM)** | 0.79148 | **0.80977** | **0.79880** |

---

## 📂 Repository Structure

```text
climate-risk-health-prediction/
├── data/
│   ├── Train.csv                     # 3,146 training records
│   ├── Test.csv                      # 1,030 test records
│   ├── climate_features.csv          # 4,176 climate records (CHIRPS, ERA5, MODIS, SRTM)
│   ├── SampleSubmission.csv          # Submission format template
│   ├── data_dictionary.csv           # Dictionary for main attributes
│   └── downloaded_climate_features_data_dictionary.csv
├── src/
│   ├── utils.py                      # Multi-metric scorer & reproducibility seed utilities
│   ├── features.py                   # Domain feature extraction pipeline
│   ├── models.py                     # LightGBM, XGBoost, and Logistic Regression model factory
│   ├── train.py                      # Stratified K-Fold cross-validation & prediction engine
│   ├── ensemble.py                   # Nelder-Mead optimal weight blending
│   └── run_pipeline.py               # Automated end-to-end multi-model runner
├── submissions/
│   ├── sub_ensemble_cv_0.7988.csv    # 🥇 Top verified ensemble submission (CV: 0.7988)
│   ├── sub_xgb_cv_0.7987.csv         # 🥈 Best single model: XGBoost (CV: 0.7987)
│   ├── sub_lgbm_cv_0.7978.csv        # 🥉 LightGBM (CV: 0.7978)
│   └── sub_logistic_cv_0.7799.csv    # Calibrated baseline
├── README.md                         # Documentation and benchmarks
└── requirements.txt                  # Python dependencies
```

---

## 🚀 Getting Started

### 1. Installation
```bash
cd climate-risk-health-prediction
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
```bash
python src/run_pipeline.py
```
This script will:
1. Merge the demographic dataset with satellite climate features on `ID`.
2. Extract all 78 engineered features.
3. Run 5-fold stratified cross-validation on Logistic Regression, XGBoost, and LightGBM.
4. Optimize ensemble weights using Nelder-Mead simplex search.
5. Generate verified submission files in `submissions/`.

### 3. Verify Submission Compliance
Ensure all competition rules are met:
```bash
python -c "
import pandas as pd
sub = pd.read_csv('submissions/sub_ensemble_cv_0.7988.csv')
sample = pd.read_csv('data/SampleSubmission.csv')
assert len(sub) == len(sample)
assert list(sub.columns) == ['ID', 'TargetF1', 'TargetRAUC']
assert (sub['TargetF1'] == (sub['TargetRAUC'] >= 0.5).astype(int)).all()
print('Submission format 100% compliant with Zindi rules!')
"
```
