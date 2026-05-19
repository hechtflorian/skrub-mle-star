
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--default-pip"])
    except subprocess.CalledProcessError:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import lightgbm as lgb
except ImportError:
    install_package('lightgbm')
    import lightgbm as lgb

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
train_data = pd.read_csv('./input/train.csv')
test_data = pd.read_csv('./input/test.csv')

# Prepare features and target
X = train_data.drop('median_house_value', axis=1)
y = train_data['median_house_value']
test_features = test_data

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the LightGBM model
model = lgb.LGBMRegressor(objective='regression', n_estimators=1000)
model.fit(X_train, y_train)

# Predict and evaluate on validation set
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Final Validation Performance: {rmse}")

# Predict on test data
test_predictions = model.predict(test_features)

# Save predictions to file (if required)
np.savetxt('submission.csv', test_predictions, fmt='%f', header='median_house_value', comments='')
