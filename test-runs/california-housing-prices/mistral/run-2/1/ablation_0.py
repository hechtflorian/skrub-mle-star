
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

if not os.path.exists(train_path):
    raise FileNotFoundError(f"Error: The training data file 'train.csv' was not found in the '{input_dir}' directory.")

train = pd.read_csv(train_path)
X_train = train.drop("median_house_value", axis=1)
y_train = train["median_house_value"]

# Base setup
imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_train = X_train_imputed

def add_features(df):
    df["rooms_per_household"] = df["total_rooms"] / np.where(df["households"] == 0, 1, df["households"])
    df["bedrooms_per_room"] = df["total_bedrooms"] / np.where(df["total_rooms"] == 0, 1, df["total_rooms"])
    df["population_per_household"] = df["population"] / np.where(df["households"] == 0, 1, df["households"])
    df["income_per_person"] = df["median_income"] / np.where(df["population"] == 0, 1, df["population"])
    df.fillna(0, inplace=True)
    return df

def evaluate_model(X, y, model, name, ablation_name=None):
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, random_state=42)
    scaler = RobustScaler()
    X_sample_scaled = scaler.fit_transform(X_sample)
    model.fit(X_sample_scaled, y_sample)
    y_pred = model.predict(X_sample_scaled)
    score = np.sqrt(mean_squared_error(y_sample, y_pred))
    print(f"{name} RMSE: {score} ({ablation_name if ablation_name else 'Base'})")
    return score

# Base performance
X_feat = add_features(X_train.copy())
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}
if CatBoostRegressor is not None:
    models["CatBoost"] = CatBoostRegressor(verbose=0, random_state=42)
if LGBMRegressor is not None:
    models["LightGBM"] = LGBMRegressor(random_state=42)

base_scores = {}
for name, model in models.items():
    base_scores[name] = evaluate_model(X_feat, y_train, model, name)

# Ablation 1: Without feature engineering
print("\n--- Ablation 1: Without Feature Engineering ---")
X_no_feat = X_train.copy()
no_feat_scores = {}
for name, model in models.items():
    no_feat_scores[name] = evaluate_model(X_no_feat, y_train, model, name, "No Feature Engineering")

# Ablation 2: Without scaling
print("\n--- Ablation 2: Without Scaling ---")
no_scaler_scores = {}
for name, model in models.items():
    X_sample, _, y_sample, _ = train_test_split(X_feat, y_train, train_size=0.1, random_state=42)
    model.fit(X_sample, y_sample)
    y_pred = model.predict(X_sample)
    score = np.sqrt(mean_squared_error(y_sample, y_pred))
    no_scaler_scores[name] = score
    print(f"{name} RMSE: {score} (No Scaling)")

# Ablation 3: Without imputation
print("\n--- Ablation 3: Without Imputation ---")
X_no_impute = train.drop("median_house_value", axis=1)
X_feat_no_impute = add_features(X_no_impute.copy())
impute_scores = {}
for name, model in models.items():
    try:
        impute_scores[name] = evaluate_model(X_feat_no_impute, y_train, model, name, "No Imputation")
    except:
        impute_scores[name] = float("inf")
        print(f"{name} RMSE: Failed (No Imputation)")

# Compare contributions
print("\n--- Performance Contribution Analysis ---")
for model in base_scores:
    delta_feat = base_scores[model] - no_feat_scores.get(model, float("inf"))
    delta_scaler = base_scores[model] - no_scaler_scores.get(model, float("inf"))
    delta_impute = base_scores[model] - impute_scores.get(model, float("inf"))

    print(f"\n{model} Contributions:")
    print(f" - Feature Engineering: {'+' if delta_feat < 0 else ''}{delta_feat:.4f}")
    print(f" - Scaling: {'+' if delta_scaler < 0 else ''}{delta_scaler:.4f}")
    print(f" - Imputation: {'+' if delta_impute < 0 else ''}{delta_impute:.4f}")
    print(f"Most impactful component: {max([('Feature Engineering', delta_feat), ('Scaling', delta_scaler), ('Imputation', delta_impute)], key=lambda x: abs(x[1]))[0]}")
