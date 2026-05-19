
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from skrub import TableVectorizer
import xgboost as xgb

# Load data
train_data = pd.read_csv('./input/train.csv')
X = train_data.drop(columns=['median_house_value'])
y = train_data['median_house_value']

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline model (original ensemble)
primary_pipeline = Pipeline([
    ("preprocessor", TableVectorizer()),
    ("regressor", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10))
])
secondary_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
ensemble_model = VotingRegressor([('rf', primary_pipeline), ('xgb', secondary_model)])
ensemble_model.fit(X_train, y_train)
val_preds_ensemble = ensemble_model.predict(X_val)
baseline_rmse = np.sqrt(mean_squared_error(y_val, val_preds_ensemble))
print(f'Baseline (Ensemble) Validation RMSE: {baseline_rmse}')

# Ablation 1: Remove TableVectorizer (use raw features)
primary_no_tv = Pipeline([
    ("regressor", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10))
])
ensemble_no_tv = VotingRegressor([
    ('rf', primary_no_tv),
    ('xgb', secondary_model)
])
ensemble_no_tv.fit(X_train, y_train)
val_preds_no_tv = ensemble_no_tv.predict(X_val)
rmse_no_tv = np.sqrt(mean_squared_error(y_val, val_preds_no_tv))
print(f'Ablation 1 (No TableVectorizer) Validation RMSE: {rmse_no_tv}')
performance_drop_tv = rmse_no_tv - baseline_rmse

# Ablation 2: Remove XGBoost (only RandomForest)
primary_pipeline.fit(X_train, y_train)
val_preds_rf_only = primary_pipeline.predict(X_val)
rmse_rf_only = np.sqrt(mean_squared_error(y_val, val_preds_rf_only))
print(f'Ablation 2 (Only RandomForest) Validation RMSE: {rmse_rf_only}')
performance_drop_xgb = rmse_rf_only - baseline_rmse

# Ablation 3: Reduce RandomForest max_depth to 5
primary_shallow_rf = Pipeline([
    ("preprocessor", TableVectorizer()),
    ("regressor", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=5))
])
shallow_ensemble = VotingRegressor([
    ('rf', primary_shallow_rf),
    ('xgb', secondary_model)
])
shallow_ensemble.fit(X_train, y_train)
val_preds_shallow = shallow_ensemble.predict(X_val)
rmse_shallow = np.sqrt(mean_squared_error(y_val, val_preds_shallow))
print(f'Ablation 3 (Shallow RandomForest) Validation RMSE: {rmse_shallow}')
performance_drop_shallow = rmse_shallow - baseline_rmse

# Compare performance drops
print("\nPerformance drops compared to baseline:")
print(f"Removing TableVectorizer: +{performance_drop_tv:.4f}")
print(f"Removing XGBoost: +{performance_drop_xgb:.4f}")
print(f"Shallow RandomForest: +{performance_drop_shallow:.4f}")

# Identify most contributing component
drops = {
    "TableVectorizer": performance_drop_tv,
    "XGBoost": performance_drop_xgb,
    "RandomForest_depth": performance_drop_shallow
}
most_important = max(drops, key=drops.get)
print(f"\nThe most contributing component is: {most_important}")
