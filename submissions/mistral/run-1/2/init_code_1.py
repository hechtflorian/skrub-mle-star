
import subprocess
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Install required packages if not installed
packages = ["lightgbm", "scikit-learn"]
for package in packages:
    try:
        __import__(package)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

import lightgbm as lgb

# Load data
train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')

# Features and target
X = train.drop('median_house_value', axis=1)
y = train['median_house_value']
X_test = test

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model with improved parameters
model = lgb.LGBMRegressor(
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
model.fit(X_train, y_train)

# Predict on validation set and evaluate
val_preds = model.predict(X_val)
rmse = mean_squared_error(y_val, val_preds) ** 0.5
print(f'Final Validation Performance: {rmse}')

# Predict on test data and save submission
test_preds = model.predict(X_test)
pd.DataFrame({'median_house_value': test_preds}).to_csv('predictions.csv', index=False)
