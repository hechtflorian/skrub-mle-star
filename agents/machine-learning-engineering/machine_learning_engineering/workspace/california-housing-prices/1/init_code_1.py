
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os

# Install required modules if not already installed
try:
    from catboost import CatBoostRegressor
except ImportError:
    os.system("pip install catboost")
    from catboost import CatBoostRegressor

try:
    import lightgbm as lgb
except ImportError:
    os.system("pip install lightgbm")
    import lightgbm as lgb

# Load data
input_dir = "./input"
train = pd.read_csv(os.path.join(input_dir, "train.csv"))
test = pd.read_csv(os.path.join(input_dir, "test.csv"))

# Check if target column exists
if "median_house_value" not in train.columns:
    raise ValueError("Target column 'median_house_value' not found in training data.")

# Features and target
X = train.drop("median_house_value", axis=1)
y = train["median_house_value"]
X_test = test.copy()

# Subsampling for faster training (if dataset is large)
sample_fraction = 0.1
if len(X) > 10000:
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_fraction, random_state=42)
else:
    X_sample, y_sample = X, y

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Model 1: RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))

# Model 2: CatBoostRegressor
cat = CatBoostRegressor(verbose=0, random_state=42)
cat.fit(X_train, y_train)
cat_pred = cat.predict(X_val)
cat_rmse = np.sqrt(mean_squared_error(y_val, cat_pred))

# Model 3: LightGBM
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
params = {
    "objective": "regression",
    "metric": "rmse",
    "random_state": 42
}
lgb_model = lgb.train(params, lgb_train, valid_sets=[lgb_val], num_boost_round=100, verbose_eval=0)
lgb_pred = lgb_model.predict(X_val)
lgb_rmse = np.sqrt(mean_squared_error(y_val, lgb_pred))

# Select the best model based on validation RMSE
models = {
    "RandomForest": rf_rmse,
    "CatBoost": cat_rmse,
    "LightGBM": lgb_rmse
}
best_model_name = min(models, key=models.get)
best_model = {
    "RandomForest": rf,
    "CatBoost": cat,
    "LightGBM": lgb_model
}[best_model_name]

print(f"Final Validation Performance: {models[best_model_name]}")

# Predict on test data
test_pred = best_model.predict(X_test)

# Save predictions
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False, header=True)
