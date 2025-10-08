"""
Decision Tree Classifier from Scratch implementing Gini Impurity and Entropy Splitting.

Author: godmode-dev
License: MIT
"""

import numpy as np
from typing import Optional, List, Dict, Union, Tuple


class Node:
    """
    Node structure for the Decision Tree.
    """

    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["Node"] = None,
        right: Optional["Node"] = None,
        *,
        value: Optional[int] = None,
        gain: Optional[float] = None
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.gain = gain

    def is_leaf_node(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier for multi-class classification.

    Parameters
    ----------
    max_depth : int, default=10
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    criterion : str, default='gini'
        The function to measure split quality ('gini' or 'entropy').
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        criterion: str = "gini"
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root: Optional[Node] = None
        self.feature_importances_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Build decision tree from training dataset."""
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=int).ravel()

        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)
        self.root = self._build_tree(X, y, depth=0)
        
        # Normalize feature importances
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp

        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find best split
        best_feat, best_thresh, best_gain = self._best_split(X, y, n_features)

        if best_gain == 0.0 or best_feat is None:
            return Node(value=self._most_common_label(y))

        # Record feature importance contribution
        self.feature_importances_[best_feat] += best_gain * n_samples

        # Split children recursively
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(
            feature=best_feat,
            threshold=best_thresh,
            left=left_child,
            right=right_child,
            gain=best_gain
        )

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, n_features: int
    ) -> Tuple[Optional[int], Optional[float], float]:
        best_gain = -1.0
        split_feat, split_thresh = None, None

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thresh in thresholds:
                gain = self._information_gain(y, X_column, thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_feat = feat_idx
                    split_thresh = thresh

        return split_feat, split_thresh, max(best_gain, 0.0)

    def _information_gain(self, y: np.ndarray, X_column: np.ndarray, threshold: float) -> float:
        parent_impurity = self._calculate_impurity(y)

        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0.0

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        imp_l, imp_r = self._calculate_impurity(y[left_idxs]), self._calculate_impurity(y[right_idxs])
        child_impurity = (n_l / n) * imp_l + (n_r / n) * imp_r

        return parent_impurity - child_impurity

    def _split(self, X_column: np.ndarray, split_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _calculate_impurity(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)

        if self.criterion == "entropy":
            return -np.sum(probabilities * np.log2(probabilities + 1e-15))
        else:  # Gini
            return 1.0 - np.sum(probabilities ** 2)

    def _most_common_label(self, y: np.ndarray) -> int:
        if len(y) == 0:
            return 0
        vals, counts = np.unique(y, return_counts=True)
        return int(vals[np.argmax(counts)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for sample array."""
        if self.root is None:
            raise ValueError("Model is not fitted yet.")

        X = np.array(X, dtype=np.float64)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x: np.ndarray, node: Node) -> int:
        if node.is_leaf_node():
            return int(node.value)

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def print_tree(self, node: Optional[Node] = None, indent: str = "  ") -> None:
        """Print ASCII representation of decision tree structure."""
        if node is None:
            node = self.root

        if node.is_leaf_node():
            print(f"{indent}Predict Class -> {node.value}")
            return

        print(f"{indent}[Feature {node.feature} <= {node.threshold:.4f}] (Gain: {node.gain:.4f})")
        print(f"{indent}├── Left:")
        self.print_tree(node.left, indent + "│   ")
        print(f"{indent}└── Right:")
        self.print_tree(node.right, indent + "    ")


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)

    clf = DecisionTreeClassifier(max_depth=4, criterion="gini")
    clf.fit(X, y)

    preds = clf.predict(X)
    acc = np.mean(preds == y)
    print(f"Training Accuracy: {acc * 100:.2f}%")
    print("\nDecision Tree Structure:")
    clf.print_tree()