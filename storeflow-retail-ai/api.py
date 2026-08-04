"""
FastAPI REST microservice serving live prediction endpoints for Storeflow AI.

Author: godmode-dev
License: MIT
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import pandas as pd

from data.generate_ghana_retail_data import generate_ghanaian_retail_dataset
from models.demand_forecaster import StoreflowDemandForecaster
from models.inventory_risk_predictor import InventoryStockoutPredictor
from models.customer_segmentation import CustomerRFMAnalyzer

app = FastAPI(
    title="Storeflow AI Predictive Engine API",
    description="REST API delivering ML revenue forecasts, stockout risk assessments, and RFM customer segmentation for Ghanaian retail.",
    version="1.0.0"
)

# Load global dataset & models
df_sales, df_inventory = generate_ghanaian_retail_dataset()
forecaster = StoreflowDemandForecaster()
forecaster.train(df_sales)
predictor = InventoryStockoutPredictor()
rfm_analyzer = CustomerRFMAnalyzer()


class ForecastRequest(BaseModel):
    days_ahead: int = Field(default=7, ge=1, le=30, example=7)


class StockoutCheckRequest(BaseModel):
    current_stock: int = Field(..., example=12)
    reorder_point: int = Field(..., example=15)
    avg_daily_sales: float = Field(..., example=4.5)
    unit_price_ghs: float = Field(..., example=120.0)


@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "service": "Storeflow AI Analytics API",
        "integration_target": "https://storeflow-by-flywheel.pages.dev"
    }


@app.post("/predict/demand", tags=["Demand Forecasting"])
def forecast_demand(req: ForecastRequest):
    df_forecast = forecaster.forecast_next_days(df_sales, days_ahead=req.days_ahead)
    return {
        "forecast_days": req.days_ahead,
        "currency": "GHS",
        "predictions": df_forecast.to_dict(orient="records")
    }


@app.post("/predict/stockout-risk", tags=["Inventory Analytics"])
def check_stockout_risk(req: StockoutCheckRequest):
    days_left = round(req.current_stock / (req.avg_daily_sales + 1e-5), 1)
    if days_left <= 3 or req.current_stock <= (req.reorder_point * 0.5):
        risk = "CRITICAL"
        reorder_units = max(10, int(req.avg_daily_sales * 14 - req.current_stock))
    elif days_left <= 7 or req.current_stock <= req.reorder_point:
        risk = "HIGH"
        reorder_units = max(10, int(req.avg_daily_sales * 14 - req.current_stock))
    else:
        risk = "NORMAL"
        reorder_units = 0

    est_cost = round(reorder_units * req.unit_price_ghs * 0.75, 2)

    return {
        "stockout_risk_tier": risk,
        "days_remaining": days_left,
        "suggested_reorder_units": reorder_units,
        "estimated_reorder_cost_ghs": est_cost
    }


@app.get("/analytics/rfm-summary", tags=["Customer Segmentation"])
def get_rfm_summary():
    df_rfm = rfm_analyzer.segment_customers(df_sales)
    summary = df_rfm["segment_label"].value_counts().to_dict()
    return {"rfm_segments": summary, "total_customers_analyzed": len(df_rfm)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
