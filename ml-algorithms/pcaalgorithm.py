# To transform a high-dimensional dataset into a lower-dimensional one while preserving as much variance as possible
# useful for data visualization, reducing noise, and speeding up machine learning models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

# Generate a sample dataset with 3 features and 3 distinct clusters.
# In a real-world scenario, you would load your own data here.
X, y = make_blobs(n_samples=500, n_features=3, centers=3, random_state=42)

# Print the original shape to see the number of dimensions.
print(f"Original data shape: {X.shape}")

# Instantiate the PCA model. We want to reduce to 2 principal components.
pca = PCA(n_components=2)

# Fit the model to the data and transform the data simultaneously.
X_reduced = pca.fit_transform(X)

# The new shape confirms the reduction.
print(f"Reduced data shape: {X_reduced.shape}")

# Create a scatter plot of the new, 2-dimensional data.
plt.figure(figsize=(8, 6))
# We color the points using the original labels 'y' to see the clusters clearly.
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)

# Add clear titles and labels.
plt.title('2D PCA Projection of a 3D Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()