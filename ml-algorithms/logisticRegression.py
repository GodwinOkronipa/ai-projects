"""
Logistic Regression Implementation from Scratch using Binary Cross-Entropy Gradient Descent.

Author: godmode-dev
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class LogisticRegression:
    """
    Logistic Regression binary classifier.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Gradient descent learning rate.
    iterations : int, default=1000
        Number of training iterations.
    threshold : float, default=0.5
        Decision threshold for class probability assignment.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        threshold: float = 0.5
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.threshold = threshold
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "LogisticRegression":
        """Fit binary classifier via Gradient Descent."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.iterations):
            # Linear combination: z = X * w + b
            linear_model = np.dot(X, self.weights) + self.bias
            # Sigmoid activation: y_hat = P(y = 1 | X)
            predictions = self._sigmoid(linear_model)

            # Binary Cross Entropy Loss
            eps = 1e-15
            pred_clipped = np.clip(predictions, eps, 1 - eps)
            loss = -np.mean(y * np.log(pred_clipped) + (1 - y) * np.log(1 - pred_clipped))
            self.loss_history.append(loss)

            # Gradient computation
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if verbose and i % (self.iterations // 10 or 1) == 0:
                print(f"Iteration {i:4d} | Log-Loss: {loss:.4f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability vector."""
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call 'fit()' first.")

        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels (0 or 1)."""
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate Accuracy, Precision, Recall, and F1 Score."""
        y_pred = self.predict(X)
        y_true = np.array(y, dtype=int).ravel()

        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = float((tp + tn) / (len(y_true) + 1e-10))
        precision = float(tp / (tp + fp + 1e-10))
        recall = float(tp / (tp + fn + 1e-10))
        f1 = float(2 * precision * recall / (precision + recall + 1e-10))

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

    def plot_decision_boundary(self, X: np.ndarray, y: np.ndarray, title: str = "Logistic Regression Decision Boundary") -> None:
        """Visualize decision boundary for 2D datasets."""
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=int).ravel()

        if X_arr.shape[1] != 2:
            print("Decision boundary plotting requires a 2D feature matrix.")
            return

        x_min, x_max = X_arr[:, 0].min() - 1, X_arr[:, 0].max() + 1
        y_min, y_max = X_arr[:, 1].min() - 1, X_arr[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = self.predict_proba(grid).reshape(xx.shape)

        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, probs, levels=20, cmap="Spectral", alpha=0.6)
        plt.colorbar(label="P(y = 1)")
        plt.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr, cmap="bwr", edgecolors="k", alpha=0.9)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    # Synthetic dataset
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(X, y, verbose=True)

    metrics = model.evaluate(X, y)
    print("\nClassification Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")