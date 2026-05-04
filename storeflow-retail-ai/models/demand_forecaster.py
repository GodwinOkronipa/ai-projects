"""
Time-Series Demand & GHS Revenue Forecasting Machine Learning Engine.

Author: godmode-dev
License: MIT
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class StoreflowDemandForecaster:
    """
    Predicts future GHS store revenue and item demand using lag features, rolling windows,
    day-of-week encoding, and machine learning regression.
    """

    def __init__(self, model_type: str = "xgboost") -> None:
        self.model_type = model_type
        if model_type == "xgboost" and XGBOOST_AVAILABLE:
            self.model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
        else:
            self.model = Ridge(alpha=1.0)
            
        self.is_fitted = False
        self.feature_names: List[str] = []

    def create_time_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Construct time-series lag and calendar features from daily revenue aggregations.
        """
        df = df_daily.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        df["day_of_week"] = df["date"].dt.dayofweek
        df["day_of_month"] = df["date"].dt.day
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Target variable: daily total revenue in GHS
        target = "total_revenue_ghs"

        # Create rolling window and lag features
        for lag in [1, 2, 3, 7, 14]:
            df[f"lag_{lag}"] = df[target].shift(lag)

        df["rolling_mean_7d"] = df[target].shift(1).rolling(window=7).mean()
        df["rolling_std_7d"] = df[target].shift(1).rolling(window=7).std()

        df = df.dropna().reset_index(drop=True)
        return df

    def prepare_data(self, df_sales: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Aggregate sales by date and generate feature set."""
        df_daily = df_sales.groupby("date").agg({
            "total_revenue_ghs": "sum",
            "quantity": "sum",
            "profit_ghs": "sum"
        }).reset_index()

        df_featured = self.create_time_features(df_daily)
        feature_cols = [c for c in df_featured.columns if c not in ["date", "total_revenue_ghs", "quantity", "profit_ghs"]]
        
        self.feature_names = feature_cols
        X = df_featured[feature_cols]
        y = df_featured["total_revenue_ghs"]
        
        return X, y

    def train(self, df_sales: pd.DataFrame) -> Dict[str, float]:
        """Fit model and return evaluation metrics."""
        X, y = self.prepare_data(df_sales)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        y_pred = self.model.predict(X_test)
        
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        return {"rmse": rmse, "mae": mae, "r2_score": r2}

    def forecast_next_days(self, df_sales: pd.DataFrame, days_ahead: int = 7) -> pd.DataFrame:
        """
        Generate recursive multi-step future daily GHS revenue forecast.
        """
        if not self.is_fitted:
            self.train(df_sales)

        df_daily = df_sales.groupby("date").agg({"total_revenue_ghs": "sum"}).reset_index()
        df_daily["date"] = pd.to_datetime(df_daily["date"])
        
        history = list(df_daily["total_revenue_ghs"].values)
        last_date = df_daily["date"].max()

        future_predictions = []

        for d in range(1, days_ahead + 1):
            next_date = last_date + pd.Timedelta(days=d)
            
            # Construct row features
            lag_1 = history[-1]
            lag_2 = history[-2] if len(history) >= 2 else history[-1]
            lag_3 = history[-3] if len(history) >= 3 else history[-1]
            lag_7 = history[-7] if len(history) >= 7 else history[-1]
            lag_14 = history[-14] if len(history) >= 14 else history[-1]
            
            roll_mean = np.mean(history[-7:])
            roll_std = np.std(history[-7:]) if len(history) >= 7 else 0.0

            row_dict = {
                "day_of_week": next_date.dayofweek,
                "day_of_month": next_date.day,
                "is_weekend": 1 if next_date.dayofweek in [5, 6] else 0,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "rolling_mean_7d": roll_mean,
                "rolling_std_7d": roll_std,
            }

            X_future = pd.DataFrame([row_dict])[self.feature_names]
            pred_ghs = float(self.model.predict(X_future)[0])
            pred_ghs = max(0.0, pred_ghs)  # Ensure non-negative revenue prediction

            future_predictions.append({
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_revenue_ghs": round(pred_ghs, 2),
                "lower_bound_ghs": round(pred_ghs * 0.88, 2),
                "upper_bound_ghs": round(pred_ghs * 1.12, 2)
            })
            history.append(pred_ghs)

        return pd.DataFrame(future_predictions)
