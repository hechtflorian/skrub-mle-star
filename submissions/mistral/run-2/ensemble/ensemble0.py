
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
import warnings

try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None

warnings.filterwarnings("ignore")

input_dir = "./input"
train_path = os.path.join(input_dir, "train.csv")
test_path = os.path.join(input_dir, "test.csv")

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Error: The training data file 'train.csv' was not found in the '{input_dir}' directory.")
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Error: The test data file 'test.csv' was not found in the '{input_dir}' directory.")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X_train = train.drop("median_house_value", axis=1)
y_train = train["median_house_value"]

imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(test), columns=test.columns)

X_train = X_train_imputed
test = X_test_imputed

def add_features(df):
    df["rooms_per_household"] = df["total_rooms"] / np.where(df["households"] == 0, 1, df["households"])
    df["bedrooms_per_room"] = df["total_bedrooms"] / np.where(df["total_rooms"] == 0, 1, df["total_rooms"])
    df["population_per_household"] = df["population"] / np.where(df["households"] == 0, 1, df["households"])
    df["income_per_person"] = df["median_income"] / np.where(df["population"] == 0, 1, df["population"])
    df.fillna(0, inplace=True)
    return df

X_train = add_features(X_train)
test = add_features(test)

X_sample, X_val, y_sample, y_val = train_test_split(X_train, y_train, train_size=0.1, random_state=42)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

if CatBoostRegressor is not None:
    models["CatBoost"] = CatBoostRegressor(verbose=0, random_state=42)
else:
    print("Warning: CatBoostRegressor not installed. Skipping.")
if LGBMRegressor is not None:
    models["LightGBM"] = LGBMRegressor(random_state=42)
else:
    print("Warning: LGBMRegressor not installed. Skipping.")

scaler = RobustScaler()
X_sample_scaled = scaler.fit_transform(X_sample)
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test)

# Generate meta-features using 5-fold CV
meta_features = []
meta_test_features = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_sample_scaled, y_sample)
    val_pred = model.predict(X_val_scaled)
    score = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"{name} Validation RMSE: {score}")
    meta_features.append([])
    meta_test_features.append([])

    for train_idx, val_idx in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        meta_features[-1].extend(model.predict(X_val_fold))

    # Train on full training data for test predictions
    model.fit(X_train_scaled, y_train)
    meta_test_features[-1] = model.predict(X_test_scaled)

# Stack meta-features
meta_features = np.column_stack(meta_features)
meta_test_features = np.column_stack(meta_test_features)

# Train meta-model (Ridge Regression)
meta_model = Ridge(alpha=1.0)
meta_model.fit(meta_features, y_train)

# Cross-validated predictions for validation set
cv_pred = np.mean(meta_features, axis=1)
final_val_rmse = np.sqrt(mean_squared_error(y_train, cv_pred))
print(f"Final Validation Performance: {final_val_rmse}")

# Generate final test predictions
test_predictions = meta_model.predict(meta_test_features)

submission = pd.DataFrame({"median_house_value": test_predictions})
submission.to_csv("submission.csv", index=False, header=True)
