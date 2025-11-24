# 🧮 Machine Learning Algorithms from Scratch

This directory contains first-principles implementations of essential machine learning algorithms built using pure **NumPy** and linear algebra.

---

## 📌 Included Algorithms

| Module | Algorithm | Optimization Method | Metrics / Features |
| :--- | :--- | :--- | :--- |
| [`linearRegression.py`](./linearRegression.py) | **Linear Regression** | Gradient Descent & Closed-Form Normal Equation | $R^2$, MSE, RMSE, MAE, Fit Visualizer |
| [`logisticRegression.py`](./logisticRegression.py) | **Logistic Regression** | Binary Cross-Entropy Gradient Descent | Accuracy, Precision, Recall, F1, Decision Boundary Plot |
| [`decisionTree.py`](./decisionTree.py) | **Decision Tree Classifier** | Gini Impurity / Entropy Split | Feature Importance, ASCII Tree Visualizer |
| [`kmeansclustering.py`](./kmeansclustering.py) | **K-Means Clustering** | K-Means++ Centroid Optimization | Inertia Tracking, Cluster Visualizer |
| [`knearestneighbours.py`](./knearestneighbours.py) | **k-NN Classifier & Regressor** | Euclidean / Manhattan Distance Matching | Classification & Regression Modes |
| [`pcaalgorithm.py`](./pcaalgorithm.py) | **Principal Component Analysis** | Covariance Eigendecomposition | Explained Variance Ratio, Projection & Reconstruction |

---

## 🧪 Running Unit Tests

```bash
# Run pytest from root directory
python -m pytest ml-algorithms/tests/
```
