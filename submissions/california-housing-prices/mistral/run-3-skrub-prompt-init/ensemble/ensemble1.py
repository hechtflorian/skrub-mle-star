
# Single-file self-contained Python script for median_house_value prediction with dynamic local ensemble
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]
X_test = test_df.copy()

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline for the base model (Solution 1)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42, max_depth=6, n_estimators=300))
])

# Train the base model
pipeline.fit(X_train, y_train)

# Generate validation predictions from base model
val_predictions = pipeline.predict(X_val)
base_val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
print(f"Base Model Validation RMSE: {base_val_rmse}")

# Select features for clustering
cluster_features = ['median_income', 'housing_median_age', 'latitude', 'longitude']
X_train_cluster = X_train[cluster_features]
X_val_cluster = X_val[cluster_features]
X_test_cluster = X_test[cluster_features]

# Scale clustering features
scaler_cluster = StandardScaler()
X_train_cluster_scaled = scaler_cluster.fit_transform(X_train_cluster)
X_val_cluster_scaled = scaler_cluster.transform(X_val_cluster)
X_test_cluster_scaled = scaler_cluster.transform(X_test_cluster)

# Perform K-means clustering (k=7)
kmeans = KMeans(n_clusters=7, random_state=42)
train_clusters = kmeans.fit_predict(X_train_cluster_scaled)
val_clusters = kmeans.predict(X_val_cluster_scaled)
test_clusters = kmeans.predict(X_test_cluster_scaled)

# Calculate cluster-specific weights (inverse RMSE)
cluster_weights = {}
for cluster_id in np.unique(train_clusters):
    cluster_mask = (val_clusters == cluster_id)
    if sum(cluster_mask) > 0:
        cluster_rmse = np.sqrt(mean_squared_error(y_val[cluster_mask], val_predictions[cluster_mask]))
        cluster_weights[cluster_id] = 1 / (cluster_rmse + 1e-6)  # Add small epsilon to avoid division by zero
    else:
        cluster_weights[cluster_id] = 1.0  # Default weight if no validation samples in cluster

# Generate test predictions from base model
test_predictions = pipeline.predict(X_test)

# Apply cluster-specific weights to test predictions
weighted_test_predictions = np.zeros_like(test_predictions)
for cluster_id in np.unique(test_clusters):
    cluster_mask = (test_clusters == cluster_id)
    if cluster_id in cluster_weights:
        weighted_test_predictions[cluster_mask] = test_predictions[cluster_mask] * cluster_weights[cluster_id]

# Normalize weights to ensure they sum to 1 for each prediction
total_weights = np.zeros_like(test_clusters, dtype=float)
for cluster_id in np.unique(test_clusters):
    cluster_mask = (test_clusters == cluster_id)
    total_weights[cluster_mask] = cluster_weights[cluster_id]

weighted_test_predictions = weighted_test_predictions / total_weights

# Calculate final validation performance
val_weighted_predictions = np.zeros_like(val_predictions)
for cluster_id in np.unique(val_clusters):
    cluster_mask = (val_clusters == cluster_id)
    if cluster_id in cluster_weights:
        val_weighted_predictions[cluster_mask] = val_predictions[cluster_mask] * cluster_weights[cluster_id]

# Normalize validation weights
total_val_weights = np.zeros_like(val_clusters, dtype=float)
for cluster_id in np.unique(val_clusters):
    cluster_mask = (val_clusters == cluster_id)
    total_val_weights[cluster_mask] = cluster_weights[cluster_id]

val_weighted_predictions = val_weighted_predictions / total_val_weights
final_validation_score = np.sqrt(mean_squared_error(y_val, val_weighted_predictions))
print(f"Final Validation Performance: {final_validation_score}")

# Save predictions in the required format
output = pd.DataFrame({"median_house_value": weighted_test_predictions})
output.to_csv("submission.csv", index=False, header=True)
