"""
Customer RFM (Recency, Frequency, Monetary) Segmentation Engine.

Author: godmode-dev
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class CustomerRFMAnalyzer:
    """
    Computes Customer RFM metrics and clusters buyers into actionable business personas
    (VIP Champions, Loyal Buyers, At-Risk Customers, New Shoppers).
    """

    def __init__(self, n_clusters: int = 4) -> None:
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.scaler = StandardScaler()

    def compute_rfm(self, df_sales: pd.DataFrame) -> pd.DataFrame:
        """Calculate Recency, Frequency, and Monetary scores per customer."""
        df = df_sales.copy()
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()

        rfm = df.groupby("customer_id").agg({
            "date": lambda d: (max_date - d.max()).days,  # Recency (days since last buy)
            "transaction_id": "nunique",                  # Frequency (count of transactions)
            "total_revenue_ghs": "sum",                   # Monetary (total spent in GHS)
            "customer_phone": "first",
            "location": "first"
        }).reset_index()

        rfm.columns = ["customer_id", "recency_days", "frequency", "monetary_ghs", "customer_phone", "primary_location"]
        return rfm

    def segment_customers(self, df_sales: pd.DataFrame) -> pd.DataFrame:
        """Perform K-Means clustering on scaled RFM metrics."""
        rfm = self.compute_rfm(df_sales)
        
        rfm_features = rfm[["recency_days", "frequency", "monetary_ghs"]]
        scaled_features = self.scaler.fit_transform(rfm_features)

        clusters = self.kmeans.fit_predict(scaled_features)
        rfm["cluster"] = clusters

        # Assign human-readable segment names based on cluster centroids
        segment_map = {}
        for c in range(self.n_clusters):
            cluster_data = rfm[rfm["cluster"] == c]
            avg_m = cluster_data["monetary_ghs"].mean()
            avg_r = cluster_data["recency_days"].mean()

            if avg_m > rfm["monetary_ghs"].quantile(0.75):
                segment_map[c] = "VIP Champions"
            elif avg_r < rfm["recency_days"].median() and avg_m >= rfm["monetary_ghs"].median():
                segment_map[c] = "Loyal Frequent Shoppers"
            elif avg_r > rfm["recency_days"].quantile(0.65):
                segment_map[c] = "At-Risk / Inactive"
            else:
                segment_map[c] = "Casual / Regular Buyers"

        rfm["segment_label"] = rfm["cluster"].map(segment_map)
        return rfm
