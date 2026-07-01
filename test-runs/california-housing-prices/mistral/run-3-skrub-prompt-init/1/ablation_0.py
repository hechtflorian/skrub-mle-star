
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load data
train_df = pd.read_csv("./input/train.csv")

# Subsampling for faster iteration
sample_size = min(10000, len(train_df))
train_df = train_df.sample(sample_size, random_state=42) if len(train_df) > 1000 else train_df

# Separate features and target
X = train_df.drop("median_house_value", axis=1)
y = train_df["median_house_value"]

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline model (original pipeline)
baseline_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42, max_depth=5, n_estimators=100))
])
baseline_pipeline.fit(X_train, y_train)
baseline_pred = baseline_pipeline.predict(X_val)
baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_pred))
print(f"Baseline RMSE: {baseline_rmse}")

# Ablation 1: Remove imputation (assume no missing values)
no_impute_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42, max_depth=5, n_estimators=100))
])
try:
    no_impute_pipeline.fit(X_train, y_train)
    no_impute_pred = no_impute_pipeline.predict(X_val)
    no_impute_rmse = np.sqrt(mean_squared_error(y_val, no_impute_pred))
    print(f"RMSE without imputation: {no_impute_rmse} (change: {no_impute_rmse - baseline_rmse})")
except Exception as e:
    print(f"RMSE without imputation: Failed ({str(e)})")

# Ablation 2: Remove scaling
no_scale_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GradientBoostingRegressor(random_state=42, max_depth=5, n_estimators=100))
])
no_scale_pipeline.fit(X_train, y_train)
no_scale_pred = no_scale_pipeline.predict(X_val)
no_scale_rmse = np.sqrt(mean_squared_error(y_val, no_scale_pred))
print(f"RMSE without scaling: {no_scale_rmse} (change: {no_scale_rmse - baseline_rmse})")

# Ablation 3: Simpler model (fewer estimators)
simple_model_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42, max_depth=5, n_estimators=50))
])
simple_model_pipeline.fit(X_train, y_train)
simple_model_pred = simple_model_pipeline.predict(X_val)
simple_model_rmse = np.sqrt(mean_squared_error(y_val, simple_model_pred))
print(f"RMSE with simpler model (n_estimators=50): {simple_model_rmse} (change: {simple_model_rmse - baseline_rmse})")

# Conclusion
print("\nPerformance impact summary:")
deltas = {
    "imputation": no_impute_rmse - baseline_rmse if 'no_impute_rmse' in locals() else np.inf,
    "scaling": no_scale_rmse - baseline_rmse,
    "model_complexity": simple_model_rmse - baseline_rmse
}
most_important = min(deltas, key=deltas.get)
print(f"The {most_important} component contributes most to performance (largest positive impact when present).")
