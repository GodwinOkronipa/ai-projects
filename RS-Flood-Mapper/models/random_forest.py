# models/random_forest.py

from typing import Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def prepare_data_for_rf(features, labels):
    """
    Flattens the image data into a format suitable for scikit-learn.
    (height, width, channels) -> (num_pixels, num_features)

    Args:
        features (numpy.ndarray): A 3D array of features (height, width, channels).
        labels (numpy.ndarray): A 2D array of labels (height, width).

    Returns:
        tuple: (X, y) where X is the feature array and y is the label array.
    """
    y = labels.flatten()
    num_pixels = features.shape[0] * features.shape[1]
    num_features = features.shape[2]
    X = features.reshape(num_pixels, num_features)
    return X, y


def train_random_forest(X_train, y_train, n_estimators: int = 50):
    """
    Trains a Random Forest classifier.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.
        n_estimators (int): Number of trees.

    Returns:
        RandomForestClassifier: The trained model.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=12, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model


def predict_with_rf(model, features, original_shape):
    """
    Makes a prediction on the full dataset and reshapes it back to the image dimensions.

    Args:
        model (RandomForestClassifier): The trained model.
        features (numpy.ndarray): The full feature set (height, width, channels).
        original_shape (tuple): The original (height, width) of the image.

    Returns:
        numpy.ndarray: The prediction map with the original image shape.
    """
    num_pixels = features.shape[0] * features.shape[1]
    num_features = features.shape[2]
    X_full = features.reshape(num_pixels, num_features)
    
    prediction_flat = model.predict(X_full)
    prediction_map = prediction_flat.reshape(original_shape)
    return prediction_map


def get_default_rf_model(num_features: int = 4) -> RandomForestClassifier:
    """
    Generate and train a quick baseline Random Forest model on synthetic remote sensing
    water & dry spectral samples.
    """
    np.random.seed(42)
    n_samples = 4000
    
    # 50% water samples, 50% non-water samples
    n_water = n_samples // 2
    n_land = n_samples - n_water

    # Water spectral signature:
    # Low Red, Medium Green, High Blue, High NDWI (> 0.2)
    water_r = np.random.normal(0.12, 0.05, (n_water, 1))
    water_g = np.random.normal(0.25, 0.08, (n_water, 1))
    water_b = np.random.normal(0.48, 0.10, (n_water, 1))
    water_ndwi = np.random.normal(0.55, 0.15, (n_water, 1))
    X_water = np.hstack([water_r, water_g, water_b, water_ndwi])[:, :num_features]
    y_water = np.ones(n_water, dtype=int)

    # Land / Urban / Vegetation spectral signature:
    # Higher Red, Green vegetation or soil, Low Blue, Low/Negative NDWI (< 0.0)
    land_r = np.random.normal(0.40, 0.12, (n_land, 1))
    land_g = np.random.normal(0.52, 0.12, (n_land, 1))
    land_b = np.random.normal(0.28, 0.08, (n_land, 1))
    land_ndwi = np.random.normal(-0.35, 0.18, (n_land, 1))
    X_land = np.hstack([land_r, land_g, land_b, land_ndwi])[:, :num_features]
    y_land = np.zeros(n_land, dtype=int)

    X = np.vstack([X_water, X_land])
    y = np.concatenate([y_water, y_land])

    model = train_random_forest(X, y, n_estimators=40)
    return model