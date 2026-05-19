
import subprocess
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from skrub import TableVectorizer

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install required packages
install_package('skrub')
install_package('xgboost')

# Load data
train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')

X = train_data.drop(columns=['median_house_value'])
y = train_data['median_house_value']

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build primary pipeline with TableVectorizer and RandomForest
primary_pipeline = Pipeline([
    ("preprocessor", TableVectorizer()),
    ("regressor", RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10))
])

# Build secondary model with XGBoost

import xgboost as xgb
secondary_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    random_state=42
)


# Build ensemble model
ensemble_model = VotingRegressor([
    ('rf', primary_pipeline),
    ('xgb', secondary_model)
])

# Prepare data for ensemble
# Primary pipeline handles its own preprocessing via TableVectorizer
X_train_ensemble = X_train
X_val_ensemble = X_val

# Train ensemble model
ensemble_model.fit(X_train_ensemble, y_train)

# Train individual models for validation
primary_pipeline.fit(X_train, y_train)
secondary_model.fit(
    primary_pipeline.named_steps['preprocessor'].transform(X_train),
    y_train
)

# Validate individual models
val_preds_rf = primary_pipeline.predict(X_val)
val_preds_xgb = secondary_model.predict(
    primary_pipeline.named_steps['preprocessor'].transform(X_val)
)

val_rmse_rf = np.sqrt(mean_squared_error(y_val, val_preds_rf))
val_rmse_xgb = np.sqrt(mean_squared_error(y_val, val_preds_xgb))

# Validate ensemble
val_preds_ensemble = ensemble_model.predict(X_val_ensemble)
final_validation_score = np.sqrt(mean_squared_error(y_val, val_preds_ensemble))

print(f'Random Forest Validation RMSE: {val_rmse_rf}')
print(f'XGBoost Validation RMSE: {val_rmse_xgb}')
print(f'Final Validation Performance: {final_validation_score}')

# Predict on test set
test_preds = ensemble_model.predict(test_data)

# Create submission
submission = pd.DataFrame({'median_house_value': test_preds})
submission.to_csv('submission.csv', index=False, header=True)
