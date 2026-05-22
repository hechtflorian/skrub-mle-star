
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
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
    df["rooms_per_household"] = df["total_rooms"] / np.where(df["households"] == 0, 1, df["households"])
    df["bedrooms_per_room"] = df["total_bedrooms"] / np.where(df["total_rooms"] == 0, 1, df["total_rooms"])
    df["population_per_household"] = df["population"] / np.where(df["households"] == 0, 1, df["households"])
    df["income_per_person"] = df["median_income"] / np.where(df["population"] == 0, 1, df["population"])
    df.fillna(0, inplace=True)
    return df

X_train = add_features(X_train)
test = add_features(test)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test)

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

kf = KFold(n_splits=5, shuffle=True, random_state=42)
model_scores = {}
oof_predictions = np.zeros(len(X_train))
test_predictions = {}

for name, model in models.items():
    fold_scores = []
    fold_oof_preds = np.zeros(len(X_train))

    for train_idx, val_idx in kf.split(X_train_scaled):
        X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        val_preds = model.predict(X_val_fold)
        fold_oof_preds[val_idx] = val_preds
        fold_score = np.sqrt(mean_squared_error(y_val_fold, val_preds))
        fold_scores.append(fold_score)

    model_scores[name] = np.mean(fold_scores)
    oof_predictions += fold_oof_preds / len(models)
    print(f"{name} CV RMSE: {model_scores[name]}")

    # Train on full training set and predict on test
    model.fit(X_train_scaled, y_train)
    test_predictions[name] = model.predict(X_test_scaled)

# Calculate weights
epsilon = 1e-6
weights = {name: 1 / (score + epsilon) for name, score in model_scores.items()}
total_weight = sum(weights.values())
normalized_weights = {name: weight / total_weight for name, weight in weights.items()}

print("\nModel weights:")
for name, weight in normalized_weights.items():
    print(f"{name}: {weight:.4f}")

# Create weighted predictions
final_test_preds = np.zeros(len(test))
for name in models.keys():
    final_test_preds += test_predictions[name] * normalized_weights[name]

# Calculate final validation score
final_score = np.sqrt(mean_squared_error(y_train, oof_predictions))
print(f"\nFinal Validation Performance: {final_score}")

submission = pd.DataFrame({"median_house_value": final_test_preds})
submission.to_csv("submission.csv", index=False, header=True)
