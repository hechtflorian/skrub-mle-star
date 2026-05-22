
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Install required packages
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}. Proceeding with available models.")

# Attempt to install and import CatBoost
try:
    import catboost
except ImportError:
    install_package('catboost')
    try:
        import catboost
    except ImportError:
        print("CatBoost is not available. Skipping CatBoost model.")
        catboost = None

# Attempt to install and import LightGBM
try:
    import lightgbm
except ImportError:
    install_package('lightgbm')
    try:
        import lightgbm
    except ImportError:
        print("LightGBM is not available. Skipping LightGBM model.")
        lightgbm = None

# Load data
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")

# Feature engineering
def add_features(df):
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df

train = add_features(train)
test = add_features(test)

# Prepare data
X = train.drop("median_house_value", axis=1)
y = train["median_house_value"]
X_test = test.copy()

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, max_samples=0.8),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42, subsample=0.8),
}

if catboost is not None:
    models["CatBoost"] = catboost.CatBoostRegressor(verbose=0, random_state=42)
if lightgbm is not None:
    models["LightGBM"] = lightgbm.LGBMRegressor(random_state=42, subsample=0.8, subsample_freq=1)

# Train and validate models
best_model = None
best_score = float('inf')

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    val_pred = model.predict(X_val_scaled)
    score = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"{name} Validation RMSE: {score}")
    if score < best_score:
        best_score = score
        best_model = model

print(f"Final Validation Performance: {best_score}")

# Train best model on full data
best_model.fit(scaler.fit_transform(X), y)
test_pred = best_model.predict(X_test_scaled)

# Save predictions
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False, header=True)
