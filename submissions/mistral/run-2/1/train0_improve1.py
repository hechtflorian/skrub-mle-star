
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

    # Distance-based features (assuming coast is near longitude -122.5)
    coast_longitude = -122.5
    df["distance_to_coast"] = np.abs(df["longitude"] - coast_longitude)
    df["distance_to_urban_center"] = np.sqrt((df["longitude"] + 118.24)**2 + (df["latitude"] - 34.05)**2)  # Approx LA center

    # Neighborhood density features (using 0.1-degree radius)
    df["latitude_rounded"] = (df["latitude"] * 10).round() / 10
    df["longitude_rounded"] = (df["longitude"] * 10).round() / 10

    # Calculate localized averages
    local_stats = df.groupby(["latitude_rounded", "longitude_rounded"]).agg({
        "population_per_household": "mean",
        "income_per_person": "mean",
        "median_income": "mean",
        "households": "sum"
    }).reset_index()
    local_stats.columns = ["latitude_rounded", "longitude_rounded", "local_pop_per_hh", "local_income_per_person", "local_median_income", "local_households"]

    df = df.merge(local_stats, on=["latitude_rounded", "longitude_rounded"], how="left")
    df["local_pop_density"] = df["local_households"] / (0.1 * 0.1)  # Per 0.01 degree^2 area

    # Non-linear transformations
    df["income_decile"] = pd.qcut(df["median_income"], q=10, labels=False)
    df["age_quantile"] = pd.qcut(df["housing_median_age"], q=5, labels=False)

    # Fill remaining missing values
    df.fillna(0, inplace=True)

    # Drop temporary columns
    df.drop(["latitude_rounded", "longitude_rounded"], axis=1, inplace=True)

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
