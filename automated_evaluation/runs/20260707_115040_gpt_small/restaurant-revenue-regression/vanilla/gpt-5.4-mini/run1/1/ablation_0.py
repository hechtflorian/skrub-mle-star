import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")

# Load training data only
df = pd.read_csv(TRAIN_PATH)

# --- Base feature engineering: Open Date to year/month/day ---
def add_date_features(data):
    data = data.copy()
    data["Open Date"] = pd.to_datetime(data["Open Date"])
    for f in ["year", "month", "day"]:
        data[f"Open Date_{f}"] = getattr(data["Open Date"].dt, f)
    return data.drop(columns=["Open Date"])

df_fe = add_date_features(df)

y = df_fe["revenue"].values
X = df_fe.drop(columns=["revenue"])

cat_cols = ["City", "City Group", "Type"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

# Fixed train/validation split for fair ablation comparison
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def evaluate_model(X_train, y_train, X_valid, y_valid, cat_idx, description):
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        verbose=0,
        random_seed=42,
    )
    model.fit(X_train, y_train, cat_features=cat_idx)
    pred = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, pred))
    print(f"{description}: RMSE = {rmse:.5f}")
    return rmse

# Baseline
baseline_rmse = evaluate_model(X_train, y_train, X_valid, y_valid, cat_idx, "Baseline")

# Ablation 1: Disable Open Date feature engineering
df_no_date = df.copy()
y_no_date = df_no_date["revenue"].values
X_no_date = df_no_date.drop(columns=["revenue", "Open Date"])

cat_idx_no_date = [X_no_date.columns.get_loc(c) for c in cat_cols]

X_train_nd, X_valid_nd, y_train_nd, y_valid_nd = train_test_split(
    X_no_date, y_no_date, test_size=0.2, random_state=42
)

no_date_rmse = evaluate_model(
    X_train_nd, y_train_nd, X_valid_nd, y_valid_nd, cat_idx_no_date, "Ablation: remove Open Date features"
)

# Ablation 2: Disable categorical feature handling
# Convert categorical columns to string-coded numeric labels to remove CatBoost categorical handling
X_num = X.copy()
for c in cat_cols:
    X_num[c] = X_num[c].astype("category").cat.codes

cat_idx_disabled = []  # no categorical features passed to CatBoost

X_train_num, X_valid_num, y_train_num, y_valid_num = train_test_split(
    X_num, y, test_size=0.2, random_state=42
)

no_cat_rmse = evaluate_model(
    X_train_num, y_train_num, X_valid_num, y_valid_num, cat_idx_disabled, "Ablation: disable categorical handling")

# Compare impacts
results = {
    "Baseline": baseline_rmse,
    "No Open Date features": no_date_rmse,
    "No categorical handling": no_cat_rmse,
}

best_ablation = min(
    [("No Open Date features", no_date_rmse), ("No categorical handling", no_cat_rmse)],
    key=lambda x: x[1]
)

print("\nPerformance impact relative to baseline:")
print(f"Remove Open Date features: {no_date_rmse - baseline_rmse:+.5f}")
print(f"Disable categorical handling: {no_cat_rmse - baseline_rmse:+.5f}")

# Determine which part contributes the most
impacts = {
    "Open Date feature engineering": no_date_rmse - baseline_rmse,
    "Categorical feature handling": no_cat_rmse - baseline_rmse,
}
most_important = max(impacts.items(), key=lambda x: abs(x[1]))

print(
    f"\nMost contributing part: {most_important[0]} "
    f"(RMSE change = {most_important[1]:+.5f})"
)