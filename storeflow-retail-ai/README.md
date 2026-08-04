# 🏬 Storeflow AI — Ghanaian Retail Analytics & Predictive Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-XGBoost%20%7C%20Streamlit%20%7C%20FastAPI%20%7C%20Scikit--Learn-orange.svg)]()
[![Storeflow Live](https://img.shields.io/badge/Live%20Platform-Flywheel%20Storeflow-green.svg)](https://storeflow-by-flywheel.pages.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise machine learning platform and interactive decision-support system designed specifically for Ghanaian retail enterprises (SMEs in Accra, Kumasi, Takoradi, and Tamale). 

Directly complements and integrates with the **[Flywheel Storeflow Management Platform](https://storeflow-by-flywheel.pages.dev)**.

---

## 🌟 Core Machine Learning Capabilities

1. **📊 GHS Time-Series Demand & Revenue Forecasting**:
   - Machine Learning time-series regression (**XGBoost** / **Ridge**) predicting store sales volume and daily gross revenue in Ghanaian Cedi (GH₵).
   - Incorporates rolling 7-day/14-day statistics, lag indicators, day-of-week seasonality, and weekend surges.

2. **👥 Customer Lifetime Value (RFM) Segmentation**:
   - Unsupervised **K-Means Clustering** analyzing buyer behavior across Recency (days since last purchase), Frequency (transaction velocity), and Monetary value (total GH₵ spend).
   - Automatically segments shoppers into *VIP Champions*, *Loyal Frequent Buyers*, and *At-Risk Customers*.

3. **📦 Automated Inventory Stockout Risk Classifier**:
   - Intelligent stock predictor categorizing inventory into `CRITICAL`, `HIGH`, and `NORMAL` replenishment risk tiers.
   - Calculates recommended reorder quantities and estimates capital expenditure in GH₵.

4. **🖥️ Interactive Dark Glassmorphic Web App**:
   - Built with **Streamlit** and **Plotly**, displaying real-time financial KPI cards, dynamic forecast sliders, Mobile Money (MTN MoMo, Telecel, Vodafone Cash) vs Cash breakdowns, and region filtering.

5. **⚡ Microservice REST API**:
   - **FastAPI** web service serving live predictions for integration with web/mobile POS backends.

---

## 🏗️ Project Architecture

```
storeflow-retail-ai/
├── app.py                             # Interactive Streamlit Web Application
├── api.py                             # FastAPI REST Microservice Endpoints
├── requirements.txt                   # Dependency Specification
├── README.md                          # Technical Documentation & Guides
├── data/
│   └── generate_ghana_retail_data.py  # Synthetic Ghanaian Transaction Generator
├── models/
│   ├── demand_forecaster.py           # Time-Series Revenue Forecasting Engine
│   ├── customer_segmentation.py       # RFM K-Means Segmentation Engine
│   └── inventory_risk_predictor.py    # Stockout Classifier & Reorder System
└── tests/
    └── test_storeflow_ai.py           # Pytest Automated Test Suite
```

---

## 🚀 Quickstart & Setup

### 1. Install Dependencies
```bash
cd storeflow-retail-ai
pip install -r requirements.txt
```

### 2. Generate Synthetic Ghanaian Retail Data
```bash
python data/generate_ghana_retail_data.py
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```

### 4. Launch FastAPI REST Service
```bash
python api.py
# Access Swagger Interactive Documentation at http://localhost:8001/docs
```

### 5. Run Automated Test Suite
```bash
pytest tests/
```

---

## 🔗 Live Platform Reference
- **Storeflow Live App**: [https://storeflow-by-flywheel.pages.dev](https://storeflow-by-flywheel.pages.dev)
