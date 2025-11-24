"""
Principal Component Analysis (PCA) Dimensionality Reduction from Scratch.

Author: godmode-dev
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict


class PCA:
    """
    Principal Component Analysis (PCA) linear dimensionality reduction algorithm.

    Parameters
    ----------
    n_components : int
        Number of principal components to project dataset onto.
    """

    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCA":
        """Compute principal components and variance ratios for dataset X."""
        X = np.array(X, dtype=np.float64)
        
        # 1. Center the data by subtracting mean
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. Eigendecomposition of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # 5. Store top n_components and compute explained variance ratios
        self.components_ = eigenvectors[:, : self.n_components].T
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = (eigenvalues[: self.n_components] / (total_variance + 1e-15))

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project dataset onto lower-dimensional principal component subspace."""
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA is not fitted yet. Call 'fit()' first.")

        X = np.array(X, dtype=np.float64)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit model to X and return transformed projections."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct original feature space from lower-dimensional projection."""
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCA is not fitted yet.")

        return np.dot(X_transformed, self.components_) + self.mean_

    def plot_explained_variance(self, title: str = "PCA Explained Variance Ratio") -> None:
        """Plot Pareto chart of individual and cumulative explained variance."""
        if self.explained_variance_ratio_ is None:
            raise ValueError("Model is not fitted yet.")

        plt.figure(figsize=(8, 5))
        components_range = range(1, self.n_components + 1)
        cum_variance = np.cumsum(self.explained_variance_ratio_)

        plt.bar(components_range, self.explained_variance_ratio_, alpha=0.6, color="#3498db", label="Individual Variance")
        plt.step(components_range, cum_variance, where="mid", color="#e74c3c", linewidth=2, label="Cumulative Variance")

        plt.ylabel("Explained Variance Ratio")
        plt.xlabel("Principal Component Index")
        plt.title(title)
        plt.xticks(components_range)
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    print("PCA Component Projection Shape:", X_reduced.shape)
    print(f"Explained Variance Ratios: {pca.explained_variance_ratio_}")
    print(f"Total Variance Preserved: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")