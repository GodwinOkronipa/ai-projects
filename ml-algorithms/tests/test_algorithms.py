"""
Unit test suite verifying correctness, matrix dimensions, and accuracy for ML algorithms implemented from scratch.
"""

import sys
import os
import pytest
import numpy as np

# Ensure parent module directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.datasets import make_classification, make_regression, make_blobs
from linearRegression import LinearRegression
from logisticRegression import LogisticRegression
from decisionTree import DecisionTreeClassifier
from kmeansclustering import KMeans
from knearestneighbours import KNearestNeighbors
from pcaalgorithm import PCA


def test_linear_regression():
    np.random.seed(42)
    X, y = make_regression(n_samples=100, n_features=1, noise=5.0, random_state=42)
    
    model = LinearRegression(learning_rate=0.01, iterations=1000)
    model.fit(X, y)
    
    metrics = model.evaluate(X, y)
    assert metrics["r2_score"] > 0.8
    assert len(model.predict(X)) == 100


def test_linear_regression_normal_equation():
    np.random.seed(42)
    X, y = make_regression(n_samples=50, n_features=2, noise=1.0, random_state=42)
    
    model = LinearRegression()
    model.fit_normal_equation(X, y)
    
    metrics = model.evaluate(X, y)
    assert metrics["r2_score"] > 0.9


def test_logistic_regression():
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)
    
    clf = LogisticRegression(learning_rate=0.1, iterations=500)
    clf.fit(X, y)
    
    metrics = clf.evaluate(X, y)
    assert metrics["accuracy"] >= 0.85
    assert len(clf.predict_proba(X)) == 100


def test_decision_tree():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    
    tree = DecisionTreeClassifier(max_depth=5, criterion="gini")
    tree.fit(X, y)
    
    preds = tree.predict(X)
    acc = np.mean(preds == y)
    assert acc >= 0.90
    assert len(tree.feature_importances_) == 4


def test_kmeans_clustering():
    X, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.5, random_state=42)
    
    kmeans = KMeans(n_clusters=3, init="kmeans++")
    labels = kmeans.fit_predict(X)
    
    assert len(np.unique(labels)) == 3
    assert kmeans.cluster_centers_.shape == (3, 2)
    assert kmeans.inertia_ > 0


def test_k_nearest_neighbors():
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    
    knn = KNearestNeighbors(k=3, metric="euclidean")
    knn.fit(X, y)
    
    metrics = knn.evaluate(X, y)
    assert metrics["accuracy"] >= 0.85


def test_pca():
    X = np.random.randn(100, 5)
    
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    
    assert X_reduced.shape == (100, 2)
    assert len(pca.explained_variance_ratio_) == 2
    assert np.sum(pca.explained_variance_ratio_) <= 1.0
