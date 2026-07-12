"""
Unit tests for Storeflow AI data generation, time-series forecasting, RFM segmentation, and stockout predictor.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np

# Ensure parent module directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_ghana_retail_data import generate_ghanaian_retail_dataset
from models.demand_forecaster import StoreflowDemandForecaster
from models.customer_segmentation import CustomerRFMAnalyzer
from models.inventory_risk_predictor import InventoryStockoutPredictor


@pytest.fixture(scope="module")
def dataset():
    df_sales, df_inventory = generate_ghanaian_retail_dataset(num_days=60)
    return df_sales, df_inventory


def test_dataset_generation(dataset):
    df_sales, df_inventory = dataset
    assert len(df_sales) > 0
    assert "total_revenue_ghs" in df_sales.columns
    assert "cost_ghs" in df_sales.columns
    assert len(df_inventory) > 0


def test_demand_forecaster(dataset):
    df_sales, _ = dataset
    forecaster = StoreflowDemandForecaster()
    metrics = forecaster.train(df_sales)
    
    assert "rmse" in metrics
    assert "r2_score" in metrics
    
    df_forecast = forecaster.forecast_next_days(df_sales, days_ahead=7)
    assert len(df_forecast) == 7
    assert "predicted_revenue_ghs" in df_forecast.columns


def test_customer_rfm(dataset):
    df_sales, _ = dataset
    analyzer = CustomerRFMAnalyzer(n_clusters=3)
    df_rfm = analyzer.segment_customers(df_sales)
    
    assert "segment_label" in df_rfm.columns
    assert len(df_rfm) > 0
    assert "recency_days" in df_rfm.columns


def test_inventory_predictor(dataset):
    _, df_inventory = dataset
    predictor = InventoryStockoutPredictor()
    df_analyzed = predictor.analyze_inventory(df_inventory)
    
    assert "risk_level" in df_analyzed.columns
    assert "suggested_reorder_units" in df_analyzed.columns
