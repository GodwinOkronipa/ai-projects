"""
Linear Regression Implementation from Scratch using Gradient Descent and Closed-Form Normal Equation.

Author: godmode-dev
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional


class LinearRegression:
    """
    Linear Regression model implementing both Gradient Descent and Normal Equation (Closed-form).

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size used during gradient descent parameter updates.
    iterations : int, default=1000
        Maximum number of optimization iterations.
    fit_intercept : bool, default=True
        Whether to calculate the intercept term (bias) for this model.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        fit_intercept: bool = True
    ) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: List[float] = []

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add column of ones for intercept vector calculation."""
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit model parameters using the analytical closed-form Normal Equation: θ = (X^T * X)^(-1) * X^T * y
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self.fit_intercept:
            X_b = self._add_intercept(X)
        else:
            X_b = X

        # Analytical calculation: theta = inv(X^T @ X) @ X^T @ y
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        if self.fit_intercept:
            self.bias = float(theta[0, 0])
            self.weights = theta[1:].ravel()
        else:
            self.bias = 0.0
            self.weights = theta.ravel()

        return self

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "LinearRegression":
        """
        Fit model using iterative Gradient Descent.
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.iterations):
            # Forward pass: y_hat = X * w + b
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate Mean Squared Error loss
            mse = np.mean((y - y_pred) ** 2)
            self.loss_history.append(mse)

            # Compute Gradients
            dw = (-2 / n_samples) * np.dot(X.T, (y - y_pred))
            db = (-2 / n_samples) * np.sum(y - y_pred)

            # Gradient update step
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if verbose and i % (self.iterations // 10 or 1) == 0:
                print(f"Iteration {i:4d} | MSE Loss: {mse:.4f} | Weights: {self.weights} | Bias: {self.bias:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict targets using the fitted linear model.
        """
        if self.weights is None:
            raise ValueError("Model is not fitted yet. Call 'fit()' first.")
        
        X = np.array(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Compute evaluation metrics ($R^2$ Score, MSE, RMSE, MAE).
        """
        y_pred = self.predict(X)
        y_true = np.array(y, dtype=np.float64).ravel()

        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        
        # R^2 coefficient of determination
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-10)))

        return {"mse": mse, "rmse": rmse, "mae": mae, "r2_score": r2}

    def plot_regression_line(self, X: np.ndarray, y: np.ndarray, title: str = "Linear Regression Fit") -> None:
        """Plot scatter dataset and calculated linear regression line."""
        X_arr = np.array(X, dtype=np.float64)
        y_arr = np.array(y, dtype=np.float64).ravel()

        plt.figure(figsize=(8, 5))
        plt.scatter(X_arr, y_arr, color="#2b5c8f", label="Actual Data Points", alpha=0.8, edgecolors="k")

        if X_arr.ndim == 1 or X_arr.shape[1] == 1:
            X_line = np.linspace(X_arr.min(), X_arr.max(), 100).reshape(-1, 1)
            y_line = self.predict(X_line)
            plt.plot(X_line, y_line, color="#e74c3c", linewidth=2.5, label="Fitted Regression Line")
            plt.xlabel("Feature X")
            plt.ylabel("Target y")
            plt.title(title)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.show()


if __name__ == "__main__":
    # Quick demonstration
    np.random.seed(42)
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float64).reshape(-1, 1)
    y = 2.5 * X.ravel() + 1.2 + np.random.normal(0, 0.8, size=10)

    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y, verbose=True)

    metrics = model.evaluate(X, y)
    print("\nModel Performance Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v:.4f}")