
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

# Create pipeline for the base model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42, max_depth=6, n_estimators=300))
])

# Train the model on full training data
pipeline.fit(X, y)

# Select features for clustering
cluster_features = ['median_income', 'housing_median_age', 'latitude', 'longitude']
X_train_cluster = X[cluster_features]
X_test_cluster = X_test[cluster_features]

# Scale clustering features
scaler_cluster = StandardScaler()
X_train_cluster_scaled = scaler_cluster.fit_transform(X_train_cluster)
X_test_cluster_scaled = scaler_cluster.transform(X_test_cluster)

# Perform K-means clustering (k=7)
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(X_train_cluster_scaled)
test_clusters = kmeans.predict(X_test_cluster_scaled)

# Generate test predictions from base model
test_predictions = pipeline.predict(X_test)

# Create submission directory if it doesn't exist
os.makedirs("./final", exist_ok=True)

# Save predictions in the required format
output = pd.DataFrame({"median_house_value": test_predictions})
output.to_csv("./final/submission.csv", index=False, header=True)
