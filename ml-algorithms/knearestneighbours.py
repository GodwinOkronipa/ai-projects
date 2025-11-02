"""
k-Nearest Neighbors (k-NN) Classifier & Regressor Implementation from Scratch.

Author: godmode-dev
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, Union, Dict


class KNearestNeighbors:
    """
    k-Nearest Neighbors classifier and regressor using Euclidean or Manhattan distances.

    Parameters
    ----------
    k : int, default=5
        Number of nearest neighbors to query.
    metric : str, default='euclidean'
        Distance metric ('euclidean' or 'manhattan').
    mode : str, default='classification'
        Task type ('classification' or 'regression').
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        mode: str = "classification"
    ) -> None:
        self.k = k
        self.metric = metric
        self.mode = mode
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNearestNeighbors":
        """Store training dataset samples."""
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y).ravel()
        return self

    def _compute_distances(self, X_test: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix between test array and training samples."""
        if self.metric == "manhattan":
            # |x1 - x2| sum
            return np.sum(np.abs(X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]), axis=2)
        else:  # Euclidean
            return np.sqrt(np.sum((X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2, axis=2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels or continuous values for test array X."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model is not fitted yet.")

        X_test = np.array(X, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        distances = self._compute_distances(X_test)
        predictions = []

        for row_dist in distances:
            # Find indices of k nearest neighbors
            k_indices = np.argsort(row_dist)[: self.k]
            k_nearest_labels = self.y_train[k_indices]

            if self.mode == "classification":
                # Majority voting
                vals, counts = np.unique(k_nearest_labels, return_counts=True)
                most_common = vals[np.argmax(counts)]
                predictions.append(most_common)
            else:  # Regression
                predictions.append(np.mean(k_nearest_labels))

        return np.array(predictions)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute task evaluation metrics."""
        preds = self.predict(X)
        y_true = np.array(y).ravel()

        if self.mode == "classification":
            acc = float(np.mean(preds == y_true))
            return {"accuracy": acc}
        else:
            mse = float(np.mean((preds - y_true) ** 2))
            mae = float(np.mean(np.abs(preds - y_true)))
            return {"mse": mse, "mae": mae}


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

    knn = KNearestNeighbors(k=5, metric="euclidean", mode="classification")
    knn.fit(X_train, y_train)

    metrics = knn.evaluate(X_test, y_test)
    print(f"k-NN Classification Accuracy on Iris Test Set: {metrics['accuracy'] * 100:.2f}%")