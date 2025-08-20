import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# --- Step 1: Generate a Sample Dataset ---
# We'll create a random dataset with 3 distinct clusters to work with.
# The `make_blobs` function is used for this.
X, y = make_blobs(n_samples=500, centers=3, random_state=42)

# --- Step 2: Create and Train the K-Means Model ---
# We instantiate the KMeans model and specify the number of clusters (n_clusters)
# we want to find. We'll set it to 3 since we know the data has 3 natural groups.
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')

# The `.fit()` method runs the K-Means algorithm, finding the optimal
# cluster assignments and centroids.
kmeans.fit(X)

# --- Step 3: Get the Results ---
# The model's `.labels_` attribute gives us the cluster assignment for each point.
cluster_labels = kmeans.labels_

# The `.cluster_centers_` attribute gives us the final coordinates of each centroid.
centroids = kmeans.cluster_centers_

print("Final Centroids:\n", centroids)

# --- Step 4: Visualize the Clusters ---
# Create a scatter plot of the data points. The `c=cluster_labels` part
# colors each point according to its assigned cluster.
plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.7)

# Plot the centroids on top of the data points.
# The red 'X' markers stand out to clearly show their location.
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

# Add titles and labels for clarity
plt.title('K-Means Clustering Result')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()