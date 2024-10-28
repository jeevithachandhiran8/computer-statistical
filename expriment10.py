import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Step 1: Generate synthesized data of size 200
data, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.5, random_state=42)

# Step 2: Original Data Plot
plt.figure()
plt.scatter(data[:, 0], data[:, 1], color='b')
plt.title("Original Data Plot")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 3: Standardize the features
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

# Step 4: Standardized Data Plot
plt.figure()
plt.scatter(data_standardized[:, 0], data_standardized[:, 1], color='b')
plt.title("Standardized Data Plot")
plt.xlabel("Standardized Feature 1")
plt.ylabel("Standardized Feature 2")
plt.show()

# Step 5: Elbow Graph for identifying the number of clusters
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_standardized)
    sse.append(kmeans.inertia_)

plt.figure()
plt.plot(k_range, sse, marker='o')
plt.title("Elbow Graph for identifying the number of clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# Step 6: Apply K-Means clustering with n=3 and plot the clusters
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(data_standardized)

# Step 7: Scatter plot to show the Clustering
plt.figure()
plt.scatter(data_standardized[:, 0], data_standardized[:, 1], c=clusters, cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='X', s=200, label='Centroids')
plt.title("Scatter plot to show the Clustering")
plt.xlabel("Standardized Feature 1")
plt.ylabel("Standardized Feature 2")
plt.legend()
plt.show()
