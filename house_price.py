import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.svm import SVC
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import cdist

# Load dataset
dataset = pd.read_csv("/Users/soham/Downloads/Housing.csv")

# Create binary price category based on median price
median_price = dataset['price'].median()
dataset['price_category'] = (dataset['price'] > median_price).astype(int)

# Select features and target
features = ['area', 'bedrooms', 'stories', 'parking']
x = dataset[features]
y = dataset['price_category']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train and evaluate Decision Tree classifier
dt_classifier = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
dt_classifier.fit(x_train, y_train)
y_pred_dt = dt_classifier.predict(x_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Train and evaluate Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)
y_pred_nb = nb_classifier.predict(x_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Standardize features for KNN and SVM
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

# Train and evaluate KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(x_train_scaled, y_train)
y_pred_knn = knn_classifier.predict(x_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Train and evaluate SVM classifier
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(x_train_scaled, y_train)
y_pred_svm = svm_classifier.predict(x_test_scaled)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Perform K-Means clustering
x_scaled = sc.fit_transform(x)
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(x_scaled)
kmeans_labels = kmeans.labels_
silhouette_kmeans = silhouette_score(x_scaled, kmeans_labels)

# Perform Agglomerative Clustering
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
agg_labels = agg_clustering.fit_predict(x_scaled)
silhouette_agg = silhouette_score(x_scaled, agg_labels)

# Calculate confidence scores
nb_proba = nb_classifier.predict_proba(x_test)
nb_confidence = np.mean(np.max(nb_proba, axis=1))

knn_distances, knn_indices = knn_classifier.kneighbors(x_test_scaled)
knn_mean_distance = np.mean(knn_distances, axis=1)
knn_confidence = np.mean(1 / (1 + knn_mean_distance))

svm_proba = svm_classifier.predict_proba(x_test_scaled)
svm_confidence = np.mean(np.max(svm_proba, axis=1))

kmeans_distances = np.min(cdist(x_scaled, kmeans.cluster_centers_, 'euclidean'), axis=1)
kmeans_confidences = 1 - kmeans_distances / np.max(kmeans_distances)
kmeans_avg_confidence = np.mean(kmeans_confidences)

agg_centroids = np.array([x_scaled[agg_labels == i].mean(axis=0) for i in range(n_clusters)])
agg_distances = np.min(cdist(x_scaled, agg_centroids, 'euclidean'), axis=1)
agg_confidences = 1 - agg_distances / np.max(agg_distances)
agg_avg_confidence = np.mean(agg_confidences)

# Print performance metrics
print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
print(f"Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%")
print(f"Naive Bayes Confidence: {nb_confidence:.2f}")
print(f"KNN Accuracy: {accuracy_knn * 100:.2f}%")
print(f"KNN Confidence: {knn_confidence:.2f}")
print(f"SVM Accuracy: {accuracy_svm * 100:.2f}%")
print(f"SVM Confidence: {svm_confidence:.2f}")
print(f"K-Means Silhouette Score: {silhouette_kmeans:.2f}")
print(f"K-Means Average Confidence: {kmeans_avg_confidence:.2f}")
print(f"Hierarchical Clustering Silhouette Score: {silhouette_agg:.2f}")
print(f"Hierarchical Clustering Average Confidence: {agg_avg_confidence:.2f}")

# Visualize model performance
models = ['DecTree', 'Naive Bayes', 'KNN', 'SVM', 'K-Means', 'Hierarchical']
accuracies = [accuracy_dt, accuracy_nb, accuracy_knn, accuracy_svm, silhouette_kmeans, silhouette_agg]
confidences = [0, nb_confidence, knn_confidence, svm_confidence, kmeans_avg_confidence, agg_avg_confidence]

fig, ax1 = plt.subplots(figsize=(10, 8))

color = 'tab:blue'
ax1.set_xlabel('Models')
ax1.set_ylabel('Accuracy / Silhouette Score', color=color)
bars = ax1.bar(models, accuracies, color=color, alpha=0.6, label='Accuracy / Silhouette Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Confidence', color=color)
markers = ax2.plot(models, confidences, color=color, marker='o', linestyle='-', linewidth=2, markersize=8, label='Confidence')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('Model Performance Comparison')
plt.tight_layout()
plt.show()
