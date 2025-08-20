import numpy as np
import pandas as pd

# The core data structure for tree
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        # For a decision node
        self.feature_index = feature_index # The feature to split on
        self.threshold = threshold         # The value to split at
        self.left = left                   # Left child node (if condition is met)
        self.right = right                 # Right child node (if condition is not met)
        
        # For a leaf node
        self.value = value                 # The final prediction (e.g., 0 or 1)

class DecisionTreeClassifier:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth # The maximum depth of the tree to prevent overfitting
        self.root = None           # The root node of the tree

    def fit(self, X, y):
        # We start building the tree from the root
        self.root = self._build_tree(X, y, 0)

    def predict(self, X):
        # Iterate through each data point and make a prediction
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        # Walk down the tree until we reach a leaf node
        node = self.root
        while node.value is None:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _build_tree(self, X, y, current_depth):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check for stopping criteria:
        # 1. If all samples in the current node belong to the same class
        # 2. If we've reached the maximum allowed depth
        if n_labels == 1 or current_depth >= self.max_depth:
            # This is a leaf node, so the value is the most common class
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Find the best split for the current data
        best_split = self._find_best_split(X, y)
        
        # If no split improves the Gini impurity, we stop here
        if best_split is None or best_split['gini_gain'] == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Recursively build the left and right children
        left_indices = best_split['left_indices']
        right_indices = best_split['right_indices']

        left_child = self._build_tree(X[left_indices], y[left_indices], current_depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], current_depth + 1)
        
        return Node(best_split['feature_index'], best_split['threshold'], left_child, right_child)

    def _find_best_split(self, X, y):
        best_split = None
        best_gini_gain = -1

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Split the data based on the current feature and threshold
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                
                # We can't split if one side is empty
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # Calculate Gini Impurity for the split
                current_gini = self._gini_impurity(y)
                gini_gain = current_gini - self._weighted_gini_impurity(y[left_indices], y[right_indices])

                if gini_gain > best_gini_gain:
                    best_gini_gain = gini_gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'left_indices': left_indices,
                        'right_indices': right_indices,
                        'gini_gain': gini_gain
                    }
        return best_split

    def _gini_impurity(self, y):
        # Gini Impurity: a measure of how "mixed" the labels are
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities**2)
        return gini

    def _weighted_gini_impurity(self, y_left, y_right):
        n_left = len(y_left)
        n_right = len(y_right)
        total_samples = n_left + n_right
        
        gini_left = self._gini_impurity(y_left)
        gini_right = self._gini_impurity(y_right)
        
        weighted_gini = (n_left / total_samples) * gini_left + (n_right / total_samples) * gini_right
        return weighted_gini

    def _most_common_label(self, y):
        # Helper function to find the most frequent class in a set of labels
        counts = np.bincount(y)
        return np.argmax(counts)

# --- EXAMPLE ---

# Sample data
data = {'Hours Studied': [2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5],
        'Attended Review': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0],
        'Passed Exam': [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0]}

df = pd.DataFrame(data)

# Convert DataFrame to a NumPy array for our algorithm
X_data = df[['Hours Studied', 'Attended Review']].values
y_data = df['Passed Exam'].values

# Create and train the model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_data, y_data)

# Make a prediction for a new student: 5 hours studied, attended review (1)
new_student = np.array([[5, 1]])
prediction = model.predict(new_student)

print(f"The model predicts the student will {'Pass' if prediction[0] == 1 else 'Fail'}.")