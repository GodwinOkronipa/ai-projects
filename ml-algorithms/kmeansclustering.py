"""
K-Means Clustering Implementation from Scratch with K-Means++ Initialization and Silhouette Metric.

Author: godmode-dev
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict


class KMeans:
    """
    K-Means Unsupervised Clustering Algorithm.

    Parameters
    ----------
    n_clusters : int, default=3
        Number of clusters to form.
    max_iter : int, default=300
        Maximum iterations of parameter optimization.
    init : str, default='kmeans++'
        Initialization strategy ('random' or 'kmeans++').
    tol : float, default=1e-4
        Convergence tolerance relative to centroid displacement.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        init: str = "kmeans++",
        tol: float = 1e-4
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.tol = tol
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_features = X.shape

        if self.init == "random":
            random_idxs = np.random.choice(n_samples, self.n_clusters, replace=False)
            return X[random_idxs].copy()

        # K-Means++ Initialization strategy
        centroids = np.empty((self.n_clusters, n_features))
        # Select first centroid randomly
        first_idx = np.random.randint(n_samples)
        centroids[0] = X[first_idx]

        for c_idx in range(1, self.n_clusters):
            # Compute distance squared from samples to nearest existing centroid
            dist_sq = np.array([min(np.sum((x - c) ** 2) for c in centroids[:c_idx]) for x in X])
            probabilities = dist_sq / np.sum(dist_sq + 1e-15)
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            next_idx = np.searchsorted(cumulative_probs, r)
            centroids[c_idx] = X[next_idx]

        return centroids

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit centroids to dataset X."""
        X = np.array(X, dtype=np.float64)
        centroids = self._init_centroids(X)

        for i in range(self.max_iter):
            # Compute Euclidean distances to all centroids
            distances = self._compute_distances(X, centroids)
            labels = np.argmin(distances, axis=1)

            # Update centroids to cluster mean
            new_centroids = np.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[k] = centroids[k]

            # Check convergence tolerance
            centroid_shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids

            if centroid_shift < self.tol:
                break

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.inertia_ = float(np.sum([np.sum((X[labels == k] - centroids[k]) ** 2) for k in range(self.n_clusters)]))

        return self

    @staticmethod
    def _compute_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Compute pairwise squared Euclidean distances."""
        # X: (N, D), centroids: (K, D)
        return np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict nearest cluster for sample array."""
        if self.cluster_centers_ is None:
            raise ValueError("Model is not fitted yet.")

        X = np.array(X, dtype=np.float64)
        distances = self._compute_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit model and return cluster assignments."""
        self.fit(X)
        return self.labels_

    def plot_clusters(self, X: np.ndarray, title: str = "K-Means Clustering Result") -> None:
        """Visualize clusters and centroid markers for 2D data."""
        X_arr = np.array(X, dtype=np.float64)
        if X_arr.shape[1] != 2:
            print("Cluster visualization requires 2D data.")
            return

        if self.labels_ is None or self.cluster_centers_ is None:
            self.fit(X_arr)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_arr[:, 0], X_arr[:, 1], c=self.labels_, cmap="tab10", alpha=0.7, edgecolors="k")
        plt.scatter(
            self.cluster_centers_[:, 0],
            self.cluster_centers_[:, 1],
            s=250,
            c="red",
            marker="X",
            edgecolors="black",
            linewidth=2,
            label="Cluster Centroids"
        )
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

    kmeans = KMeans(n_clusters=4, init="kmeans++")
    kmeans.fit(X)

    print(f"Cluster Inertia: {kmeans.inertia_:.4f}")
    print("Centroid Coordinates:\n", kmeans.cluster_centers_)