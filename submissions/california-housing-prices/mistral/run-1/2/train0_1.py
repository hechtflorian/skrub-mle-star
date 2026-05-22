
import subprocess
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Install required packages
packages = ["catboost", "lightgbm"]
for package in packages:
    try:
        __import__(package)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import catboost as cb
import lightgbm as lgb

# Load data
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

# Feature engineering: Add engineered features
train['rooms_per_household'] = train['total_rooms'] / train['households']
train['bedrooms_per_room'] = train['total_bedrooms'] / train['total_rooms']
train['population_per_household'] = train['population'] / train['households']

test['rooms_per_household'] = test['total_rooms'] / test['households']
test['bedrooms_per_room'] = test['total_bedrooms'] / test['total_rooms']
test['population_per_household'] = test['population'] / test['households']

# Features and target
X = train.drop('median_house_value', axis=1)
y = train['median_house_value']
X_test = test

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train CatBoost model
catboost_model = cb.CatBoostRegressor(random_state=42, verbose=0)
catboost_model.fit(X_train, y_train)
catboost_val_preds = catboost_model.predict(X_val)

# Train LightGBM model
lgbm_model = lgb.LGBMRegressor(
    random_state=42,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1
)
lgbm_model.fit(X_train, y_train)
lgbm_val_preds = lgbm_model.predict(X_val)

# Ensemble predictions (simple average)
val_preds = (catboost_val_preds + lgbm_val_preds) / 2
rmse = mean_squared_error(y_val, val_preds) ** 0.5
print(f'Final Validation Performance: {rmse}')

# Generate test predictions
catboost_test_preds = catboost_model.predict(X_test)
lgbm_test_preds = lgbm_model.predict(X_test)
test_preds = (catboost_test_preds + lgbm_test_preds) / 2

# Save submission
pd.DataFrame({'median_house_value': test_preds}).to_csv('predictions.csv', index=False)
