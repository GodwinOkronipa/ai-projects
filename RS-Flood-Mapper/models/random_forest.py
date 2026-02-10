# models/random_forest.py

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
    print("Preparing data for Random Forest...")
    # Flatten the labels
    y = labels.flatten()
    
    # Flatten each feature band and stack them as columns
    num_pixels = features.shape[0] * features.shape[1]
    num_features = features.shape[2]
    X = features.reshape(num_pixels, num_features)
    
    return X, y

def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest classifier.

    Args:
        X_train (numpy.ndarray): Training features.
        y_train (numpy.ndarray): Training labels.

    Returns:
        RandomForestClassifier: The trained model.
    """
    print("Training Random Forest model...")
    # These parameters are a good starting point for an amateur project.
    # n_estimators: number of trees in the forest.
    # n_jobs=-1: use all available CPU cores.
    # random_state: for reproducibility.
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
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
    print("Making predictions with Random Forest...")
    num_pixels = features.shape[0] * features.shape[1]
    num_features = features.shape[2]
    X_full = features.reshape(num_pixels, num_features)
    
    prediction_flat = model.predict(X_full)
    
    # Reshape the flat prediction array back to the 2D image shape
    prediction_map = prediction_flat.reshape(original_shape)
    return prediction_map