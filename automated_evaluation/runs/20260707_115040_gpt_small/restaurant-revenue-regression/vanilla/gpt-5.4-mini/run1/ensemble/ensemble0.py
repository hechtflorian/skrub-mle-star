
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

df = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Keep a copy of original Id for submission
test_ids = test["Id"].copy()

# =============================================================================
# Model A: Original lightweight CatBoost with simple factorized categoricals
# =============================================================================
df_a = df.copy()
test_a = test.copy()

# Remove Open Date-derived features and keep the pipeline lightweight
df_a = df_a.drop(columns=["Open Date"])
test_a = test_a.drop(columns=["Open Date"])

# Simple numeric encoding for categorical columns
cat_cols = ["City", "City Group", "Type"]
for c in cat_cols:
    combined = pd.concat([df_a[c], test_a[c]], axis=0)
    codes, uniques = pd.factorize(combined, sort=True)
    df_a[c] = codes[: len(df_a)].astype("int32")
    test_a[c] = codes[len(df_a) :].astype("int32")

y = df_a["revenue"].values
X_a = df_a.drop(columns=["revenue"])
X_test_a = test_a.copy()

X_train_a, X_valid_a, y_train_a, y_valid_a = train_test_split(
    X_a, y, test_size=0.2, random_state=42
)

model_a = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    verbose=0,
    random_seed=42,
)

model_a.fit(X_train_a, y_train_a)
valid_pred_a = model_a.predict(X_valid_a)
rmse_a = np.sqrt(mean_squared_error(y_valid_a, valid_pred_a))

# Refit on full data
model_a.fit(X_a, y)
test_pred_a = model_a.predict(X_test_a)

# =============================================================================
# Model B: Date-aware / frequency-encoded CatBoost variant
# =============================================================================
df_b = df.copy()
test_b = test.copy()

# Date features
for d in [df_b, test_b]:
    d["Open Date"] = pd.to_datetime(d["Open Date"])
    d["OpenYear"] = d["Open Date"].dt.year
    d["OpenMonth"] = d["Open Date"].dt.month
    d["OpenDay"] = d["Open Date"].dt.day
    d["OpenDayOfWeek"] = d["Open Date"].dt.dayofweek
    # Use a stable reference date for years since opening
    ref_date = pd.Timestamp("2024-01-01")
    d["DaysSinceOpen"] = (ref_date - d["Open Date"]).dt.days.astype(np.int32)
    d.drop(columns=["Open Date"], inplace=True)

# Frequency encoding for categorical features
cat_cols_b = ["City", "City Group", "Type"]
for c in cat_cols_b:
    freq = pd.concat([df_b[c], test_b[c]], axis=0).value_counts(dropna=False)
    df_b[c + "_freq"] = df_b[c].map(freq).astype("float32")
    test_b[c + "_freq"] = test_b[c].map(freq).astype("float32")

    # Also keep factorized version to stay CatBoost-friendly and aligned
    combined = pd.concat([df_b[c], test_b[c]], axis=0)
    codes, uniques = pd.factorize(combined, sort=True)
    df_b[c] = codes[: len(df_b)].astype("int32")
    test_b[c] = codes[len(df_b) :].astype("int32")

# Ensure no missing values after frequency encoding
for c in ["City_freq", "City Group_freq", "Type_freq"]:
    df_b[c] = df_b[c].fillna(0).astype("float32")
    test_b[c] = test_b[c].fillna(0).astype("float32")

y_b = df_b["revenue"].values
X_b = df_b.drop(columns=["revenue"])
X_test_b = test_b.copy()

X_train_b, X_valid_b, y_train_b, y_valid_b = train_test_split(
    X_b, y_b, test_size=0.2, random_state=42
)

model_b = CatBoostRegressor(
    iterations=3500,
    learning_rate=0.03,
    depth=7,
    loss_function="RMSE",
    verbose=0,
    random_seed=42,
)

model_b.fit(X_train_b, y_train_b)
valid_pred_b = model_b.predict(X_valid_b)
rmse_b = np.sqrt(mean_squared_error(y_valid_b, valid_pred_b))

# Refit on full data
model_b.fit(X_b, y_b)
test_pred_b = model_b.predict(X_test_b)

# =============================================================================
# Weighted blend: choose best weight using validation RMSE
# =============================================================================
best_w = 0.5
best_rmse = float("inf")

for w in np.arange(0.1, 1.0, 0.1):
    blend_valid = w * valid_pred_a + (1.0 - w) * valid_pred_b
    rmse = np.sqrt(mean_squared_error(y_valid_a, blend_valid))
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = float(w)

final_test_pred = best_w * test_pred_a + (1.0 - best_w) * test_pred_b

submission = pd.DataFrame({"Id": test_ids, "Prediction": final_test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Model A Validation RMSE: {rmse_a}")
print(f"Model B Validation RMSE: {rmse_b}")
print(f"Best Ensemble Weight (Model A): {best_w}")
print(f"Final Validation Performance: {best_rmse}")
