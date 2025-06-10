import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, confusion_matrix
import plotly.express as px

# Part 1: Exploratory Data Analysis
# Loading the dataset
base = pd.read_csv('covid_related_disease_data.csv', sep=',', header=0)

# Displaying basic information about the dataset
print("Dataset Shape:", base.shape)
print("\nFirst 5 rows of the dataset:")
print(base.head())
print("\nStatistical Summary:")
print(base.describe(include='all'))

# Part 2: Data Preprocessing
# Keeping only specified columns: Preexisting_Condition and Severity
base = base[['Preexisting_Condition', 'Severity']]

# Checking for missing values
print("\nMissing Values in Dataset:")
print(base.isnull().any(axis=1).sum(), "rows with missing values")
# No missing values found in the specified columns, so no need for dropna() or imputation

# Encoding categorical variables
le_condition = LabelEncoder()
le_severity = LabelEncoder()
base['Preexisting_Condition'] = le_condition.fit_transform(base['Preexisting_Condition'])
base['Severity'] = le_severity.fit_transform(base['Severity'])
print("\nEncoded DataFrame Head:")
print(base.head())

# Normalizing the data (optional, as both columns are categorical, but applying for completeness)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(base)

# Part 3: K-Means Clustering
# Step 3.1: Elbow Method to choose the number of clusters
inertia = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(8, 6))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Explanation: The elbow plot shows a bend around K=3, suggesting 3 clusters as optimal.

# Step 3.2: Training K-Means with K=3
kmeans = KMeans(n_clusters=3, random_state=42)
base['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 3.3: Visualizing clusters with PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centroids_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=base['Cluster'], cmap='viridis', label='Data Points')
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=200, linewidths=3, label='Centroids')
plt.title('Clusters with Centroids (PCA 2D Projection)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()

# Step 3.4: Clustering Performance Metrics
silhouette_avg = silhouette_score(X_scaled, base['Cluster'])
print("\nSilhouette Score for K=3:", silhouette_avg)
print("Inertia for K=3:", kmeans.inertia_)

# Explanation of K-Means:
# K-Means is an unsupervised clustering algorithm that partitions data into K clusters by:
# 1. Randomly initializing K centroids.
# 2. Assigning each data point to the nearest centroid based on Euclidean distance.
# 3. Updating centroids as the mean of assigned points.
# 4. Repeating steps 2-3 until convergence (centroids stabilize).
# The algorithm minimizes intra-cluster variance (inertia). Here, we used K=3 based on the elbow method.

# Part 4: Confusion Matrix
# Provided predicted and test labels
y_pred = [0, 3, 1, 2, 0, 3, 0, 1, 2, 0, 4, 1, 2]
y_test = [0, 3, 2, 1, 0, 4, 1, 1, 2, 0, 4, 0, 2]

# Creating confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualizing confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
