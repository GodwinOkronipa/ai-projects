import numpy as np
import pandas as pd


def extract_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    Feature engineering pipeline for climate risk mortality prediction.
    Enriches tabular demographic and environmental data with:
    - Temporal / seasonal / cyclical indicators
    - Climate and weather anomaly deltas vs short (7d), medium (30d), and seasonal (90d) baselines
    - Vulnerability and demographic interactions
    - Geospatial and administrative hierarchy extractions
    """
    out = df.copy()

    # 1. Date and Temporal Features
    out["deathdate"] = pd.to_datetime(out["deathdate"], errors="coerce")
    out["year"] = out["deathdate"].dt.year
    out["month"] = out["deathdate"].dt.month
    out["day"] = out["deathdate"].dt.day
    out["day_of_year"] = out["deathdate"].dt.dayofyear
    out["quarter"] = out["deathdate"].dt.quarter
    out["week_of_year"] = out["deathdate"].dt.isocalendar().week.astype(int)

    # Cyclical representations
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["dayofyear_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["dayofyear_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    # East Africa (Uganda) bimodal rainfall seasonality
    # Long rainy season: March (3) to May (5)
    # Short rainy season: Sept (9) to Nov (11)
    out["is_long_rain_season"] = out["month"].isin([3, 4, 5]).astype(int)
    out["is_short_rain_season"] = out["month"].isin([9, 10, 11]).astype(int)
    out["is_dry_season"] = out["month"].isin([12, 1, 2, 6, 7, 8]).astype(int)

    # 2. Temperature Anomalies and Deltas
    out["temp_range_daily"] = out["max_temperature"] - out["min_temperature"]
    out["temp_diff_7d"] = out["avg_temperature"] - out["tavg_7d"]
    out["temp_diff_30d"] = out["avg_temperature"] - out["tavg_30d"]
    out["temp_diff_90d"] = out["avg_temperature"] - out["tavg_90d"]
    out["tmax_excess_30d"] = out["max_temperature"] - out["tmax_30d"]
    out["tmin_deficit_30d"] = out["min_temperature"] - out["tmin_30d"]
    out["temp_range_anomaly"] = out["temp_range_daily"] - out["temp_range_mean_30d"]
    out["is_hot_day_current"] = (out["max_temperature"] >= 30.0).astype(int)

    # 3. Precipitation & Hydrological Anomalies
    out["rain_avg_daily_30d"] = out["rain_sum_30d"] / 30.0
    out["rain_diff_30d"] = out["precipitation"] - out["rain_avg_daily_30d"]
    out["rain_ratio_7d_30d"] = out["rain_sum_7d"] / (out["rain_sum_30d"] + 1e-4)
    out["rain_ratio_30d_90d"] = out["rain_sum_30d"] / (out["rain_sum_90d"] + 1e-4)
    out["is_rainy_day_current"] = (out["precipitation"] > 0.1).astype(int)
    out["is_heavy_rain_current"] = (out["precipitation"] >= 10.0).astype(int)
    out["rain_day_ratio_30d"] = out["rain_days_30d"] / 30.0

    # 4. Vegetation / NDVI Dynamics
    out["ndvi_diff"] = out["ndvi_30d"] - out["ndvi_90d"]
    out["ndvi_ratio"] = (out["ndvi_30d"] + 1e-4) / (out["ndvi_90d"] + 1e-4)

    # 5. Vulnerability & Demographic Interactions
    out["is_infant_child"] = (out["age"] <= 5).astype(int)
    out["is_elderly"] = (out["age"] >= 60).astype(int)
    out["is_vulnerable_age"] = ((out["age"] <= 5) | (out["age"] >= 60)).astype(int)

    out["vuln_x_hot_days"] = out["is_vulnerable_age"] * out["hot_days_30d"]
    out["vuln_x_temp_range"] = out["is_vulnerable_age"] * out["temp_range_daily"]
    out["vuln_x_temp_diff_30d"] = out["is_vulnerable_age"] * out["temp_diff_30d"]
    out["vuln_x_rain_sum_30d"] = out["is_vulnerable_age"] * out["rain_sum_30d"]
    out["vuln_x_rain_days_30d"] = out["is_vulnerable_age"] * out["rain_days_30d"]

    # 6. Geography & Topography
    out["elevation_x_slope"] = out["elevation"] * out["slope"]
    out["elevation_km"] = out["elevation"] / 1000.0

    # Parse location string (e.g. "Izimba, Iganga, Uganda")
    if "location" in out.columns:
        loc_split = out["location"].astype(str).str.split(",", expand=True)
        out["village"] = loc_split[0].str.strip()
        if loc_split.shape[1] > 1:
            out["district"] = loc_split[1].str.strip()
        else:
            out["district"] = "Unknown"

    # Drop raw date column
    out = out.drop(columns=["deathdate"])

    return out


def prepare_datasets(data_dir: str):
    """
    Load Train.csv, Test.csv, and climate_features.csv, merge by ID,
    and apply the feature engineering pipeline.
    """
    import os

    train_path = os.path.join(data_dir, "Train.csv")
    test_path = os.path.join(data_dir, "Test.csv")
    climate_path = os.path.join(data_dir, "climate_features.csv")

    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    climate_raw = pd.read_csv(climate_path)

    # Avoid duplicate deathdate column during merge
    climate_cols = [c for c in climate_raw.columns if c != "deathdate" or c == "ID"]
    climate_raw_subset = climate_raw[climate_cols]

    # Merge on ID
    train_merged = train_raw.merge(climate_raw_subset, on="ID", how="left")
    test_merged = test_raw.merge(climate_raw_subset, on="ID", how="left")

    # Extract features
    train_feat = extract_features(train_merged, is_train=True)
    test_feat = extract_features(test_merged, is_train=False)

    return train_feat, test_feat
