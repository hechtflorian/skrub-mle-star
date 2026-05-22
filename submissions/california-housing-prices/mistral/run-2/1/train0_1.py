
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
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

X = train.drop("median_house_value", axis=1)
y = train["median_house_value"]

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(test), columns=test.columns)

X = X_imputed
test = X_test_imputed

def add_features(df):
    df["rooms_per_household"] = df["total_rooms"] / np.where(df["households"] == 0, 1, df["households"])
    df["bedrooms_per_room"] = df["total_bedrooms"] / np.where(df["total_rooms"] == 0, 1, df["total_rooms"])
    df["population_per_household"] = df["population"] / np.where(df["households"] == 0, 1, df["households"])
    df["income_per_person"] = df["median_income"] / np.where(df["population"] == 0, 1, df["population"])
    df.fillna(0, inplace=True)
    return df

X = add_features(X)
test = add_features(test)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, max_samples=0.8),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42, subsample=0.8),
}

if CatBoostRegressor is not None:
    models["CatBoost"] = CatBoostRegressor(verbose=0, random_state=42)
if LGBMRegressor is not None:
    models["LightGBM"] = LGBMRegressor(random_state=42, subsample=0.8, subsample_freq=1)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test)

best_model = None
best_score = float("inf")

model_performances = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_val_scaled)
    score = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"{name} Validation RMSE: {score}")
    model_performances[name] = score
    if score < best_score:
        best_score = score
        best_model = model

ensembled_models = []
for name, model in models.items():
    if model_performances[name] < (best_score * 1.05):
        ensembled_models.append((name, model))

voting_regressor = VotingRegressor(estimators=ensembled_models)
voting_regressor.fit(X_train_scaled, y_train)
y_pred_ensemble = voting_regressor.predict(X_val_scaled)
ensemble_score = np.sqrt(mean_squared_error(y_val, y_pred_ensemble))
print(f"Ensemble Validation RMSE: {ensemble_score}")

if ensemble_score < best_score:
    best_score = ensemble_score
    best_model = voting_regressor
    print("Ensemble model selected as best model.")
else:
    print("Single best model selected.")

print(f"Final Validation Performance: {best_score}")

best_model.fit(scaler.fit_transform(X), y)
test_predictions = best_model.predict(X_test_scaled)

submission = pd.DataFrame({"median_house_value": test_predictions})
submission.to_csv("submission.csv", index=False, header=True)
