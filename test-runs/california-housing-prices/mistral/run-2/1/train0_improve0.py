
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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

X_train = train.drop("median_house_value", axis=1)
y_train = train["median_house_value"]

imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(test), columns=test.columns)

X_train = X_train_imputed
test = X_test_imputed


def add_features(df):
    # Original features
    df["rooms_per_household"] = df["total_rooms"] / np.where(df["households"] == 0, 1, df["households"])
    df["bedrooms_per_room"] = df["total_bedrooms"] / np.where(df["total_rooms"] == 0, 1, df["total_rooms"])
    df["population_per_household"] = df["population"] / np.where(df["households"] == 0, 1, df["households"])
    df["income_per_person"] = df["median_income"] / np.where(df["population"] == 0, 1, df["population"])

    # New domain-relevant features
    df["age_of_house"] = df["housing_median_age"]
    df["income_per_household"] = df["median_income"] / np.where(df["households"] == 0, 1, df["households"])

    # Interaction terms
    df["rooms_x_income"] = df["total_rooms"] * df["median_income"]

    # Log transformations for skewed features
    df["log_population"] = np.log1p(df["population"])
    df["log_total_rooms"] = np.log1p(df["total_rooms"])
    df["log_households"] = np.log1p(df["households"])

    # Handle potential NaN/Inf values
    df.fillna(0, inplace=True)

    return df


X_train = add_features(X_train)
test = add_features(test)

X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)

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
X_test_scaled = scaler.transform(test)

best_model = None
best_score = float("inf")

for name, model in models.items():
    model.fit(X_sample_scaled, y_sample)
    y_pred = model.predict(X_sample_scaled)
    score = np.sqrt(mean_squared_error(y_sample, y_pred))
    print(f"{name} Validation RMSE: {score}")
    if score < best_score:
        best_score = score
        best_model = model

print(f"Final Validation Performance: {best_score}")

best_model.fit(X_train_scaled, y_train)
test_predictions = best_model.predict(X_test_scaled)

submission = pd.DataFrame({"median_house_value": test_predictions})
submission.to_csv("submission.csv", index=False, header=True)
