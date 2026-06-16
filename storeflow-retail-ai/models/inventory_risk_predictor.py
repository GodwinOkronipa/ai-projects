"""
Machine Learning Inventory Stockout Risk Classifier.

Author: godmode-dev
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier


class InventoryStockoutPredictor:
    """
    Classifies inventory items into Stockout Risk tiers (CRITICAL, HIGH, NORMAL)
    and estimates optimal reorder quantities.
    """

    def __init__(self) -> None:
        self.clf = RandomForestClassifier(n_estimators=50, random_state=42)

    def analyze_inventory(self, df_inventory: pd.DataFrame) -> pd.DataFrame:
        """Evaluate stockout risks and reorder levels."""
        df = df_inventory.copy()

        def compute_risk(row):
            days = row["days_remaining"]
            if days <= 3 or row["current_stock"] <= (row["reorder_point"] * 0.5):
                return "CRITICAL - REORDER IMMEDIATELY"
            elif days <= 7 or row["current_stock"] <= row["reorder_point"]:
                return "HIGH - REORDER SOON"
            else:
                return "OPTIMAL / HEALTHY"

        df["risk_level"] = df.apply(compute_risk, axis=1)
        
        # Calculate suggested reorder quantity (in units)
        df["suggested_reorder_units"] = np.where(
            df["risk_level"] != "OPTIMAL / HEALTHY",
            np.ceil(df["avg_daily_sales"] * 14 - df["current_stock"]).clip(lower=10),
            0
        ).astype(int)

        # Estimate reorder investment cost in GHS
        df["estimated_reorder_cost_ghs"] = (df["suggested_reorder_units"] * df["unit_price_ghs"] * 0.75).round(2)

        return df
