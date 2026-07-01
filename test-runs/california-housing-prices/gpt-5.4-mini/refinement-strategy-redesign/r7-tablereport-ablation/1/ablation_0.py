import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

import skrub

DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TARGET_COL = "median_house_value"

train_df = pd.read_csv(TRAIN_PATH)

# Basic cleanup
for col in train_df.columns:
    if train_df[col].dtype == "object":
        train_df[col] = pd.to_numeric(train_df[col], errors="ignore")

X = train_df.drop(columns=[TARGET_COL], errors="ignore")
y = train_df[TARGET_COL].astype(float)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

def fit_eval(features_train, features_val, y_train, y_val, model_params, variant_name):
    model = HistGradientBoostingRegressor(random_state=42, **model_params)
    model.fit(features_train, y_train)
    pred = model.predict(features_val)
    score = rmse(y_val, pred)
    print(f"Ablation[{variant_name}] RMSE: {score:.6f}")
    return score

# Baseline: original-style model, fixed parameters
baseline_params = dict(
    learning_rate=0.05,
    max_depth=8,
    max_iter=500,
)
baseline_score = fit_eval(X_train, X_val, y_train, y_val, baseline_params, "baseline_hgb")

# Ablation 1: drop one highly correlated feature (households) to reduce redundancy
X_train_drop = X_train.drop(columns=["households"], errors="ignore")
X_val_drop = X_val.drop(columns=["households"], errors="ignore")
drop_score = fit_eval(
    X_train_drop, X_val_drop, y_train, y_val, baseline_params, "drop_households"
)

# Ablation 2: add a simple ratio feature (rooms per household) and keep model fixed
def add_ratio_features(df):
    df = df.copy()
    denom = df["households"].replace(0, np.nan)
    df["rooms_per_household"] = (df["total_rooms"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

X_train_ratio = add_ratio_features(X_train)
X_val_ratio = add_ratio_features(X_val)
ratio_score = fit_eval(
    X_train_ratio, X_val_ratio, y_train, y_val, baseline_params, "add_rooms_per_household"
)

# Ablation 3: structural preprocessing change for all numeric columns using TableVectorizer
# This is a comparison against direct numeric input; here it should behave similarly,
# but it tests whether standardized DataOps-style vectorization changes performance.
vectorizer = skrub.TableVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
vec_score = fit_eval(X_train_vec, X_val_vec, y_train, y_val, baseline_params, "table_vectorizer")

results = {
    "baseline_hgb": baseline_score,
    "drop_households": drop_score,
    "add_rooms_per_household": ratio_score,
    "table_vectorizer": vec_score,
}

best_variant = min(results, key=results.get)
best_score = results[best_variant]

print("\nAblation summary:")
for name, score in results.items():
    delta = score - baseline_score
    print(f"- {name}: RMSE={score:.6f} | delta_vs_baseline={delta:+.6f}")

print(f"\nBest ablation variant: {best_variant} | RMSE: {best_score:.6f}")
print(f"Final Validation Performance: {best_score:.6f}")

# Identify which part contributed most relative to baseline
improvements = {
    "drop_households": baseline_score - drop_score,
    "add_rooms_per_household": baseline_score - ratio_score,
    "table_vectorizer": baseline_score - vec_score,
}
best_contribution = max(improvements, key=improvements.get)
print(
    f"Most helpful change: {best_contribution} "
    f"(RMSE improvement: {improvements[best_contribution]:.6f})"
)