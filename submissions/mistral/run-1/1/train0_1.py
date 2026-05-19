
import subprocess
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Install necessary packages if not available
def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install packages
install_package("xgboost")
install_package("lightgbm")
install_package("scikit-learn")

import xgboost as xgb
import lightgbm as lgb

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Features and target
X = train.drop(columns=["median_house_value"])
y = train["median_house_value"]
X_test = test.copy()

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model with early_stopping_rounds in the constructor
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=1000,
    learning_rate=0.05,
    early_stopping_rounds=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=10)

# Train LightGBM model
lgb_model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
)

# Predict on validation set
xgb_val_preds = xgb_model.predict(X_val)
lgb_val_preds = lgb_model.predict(X_val)

# Ensemble predictions (simple average)
ensemble_val_preds = (xgb_val_preds + lgb_val_preds) / 2

# Calculate RMSE
rmse = mean_squared_error(y_val, ensemble_val_preds) ** 0.5

# Print final performance
print(f'Final Validation Performance: {rmse}')

# Predict on test set and ensemble predictions
xgb_test_preds = xgb_model.predict(X_test)
lgb_test_preds = lgb_model.predict(X_test)
ensemble_test_preds = (xgb_test_preds + lgb_test_preds) / 2

# Save predictions
submission = pd.DataFrame({"median_house_value": ensemble_test_preds})
submission.to_csv("submission.csv", index=False, header=True)
