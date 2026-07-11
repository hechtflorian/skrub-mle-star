
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


# Preserve date signal with age-based features
for c in ["Open Date"]:
    df[c] = pd.to_datetime(df[c])
    test[c] = pd.to_datetime(test[c])

    # Use a consistent reference date for train/test age calculations
    ref_date = max(df[c].max(), test[c].max())

    df[f"{c}_age_days"] = (ref_date - df[c]).dt.days.astype(np.float32)
    test[f"{c}_age_days"] = (ref_date - test[c]).dt.days.astype(np.float32)

    df[f"{c}_age_months"] = (df[f"{c}_age_days"] / 30.44).astype(np.float32)
    test[f"{c}_age_months"] = (test[f"{c}_age_days"] / 30.44).astype(np.float32)

    # Reduce influence of extreme ages
    for split_df in [df, test]:
        split_df[f"{c}_age_days"] = np.clip(split_df[f"{c}_age_days"], 0, None)
        split_df[f"{c}_age_months"] = np.clip(split_df[f"{c}_age_months"], 0, None)
        split_df[f"{c}_age_log"] = np.log1p(split_df[f"{c}_age_days"]).astype(np.float32)

# Target-free frequency/count encoding for categorical columns
cat_cols = ["City", "City Group", "Type"]
combined = pd.concat([df[cat_cols], test[cat_cols]], axis=0, ignore_index=True)

for c in cat_cols:
    freq = combined[c].value_counts(dropna=False)
    df[f"{c}_freq"] = df[c].map(freq).astype(np.float32)
    test[f"{c}_freq"] = test[c].map(freq).astype(np.float32)

# Keep original date column for engineered features but drop raw date itself
df = df.drop(columns=["Open Date"])
test = test.drop(columns=["Open Date"])

y = df["revenue"].values
X = df.drop(columns=["revenue"])
X_test = test.copy()

cat_idx = [X.columns.get_loc(c) for c in cat_cols]

# Time-aware validation split if possible, otherwise a repeated seed split
if "Open Date_age_days" in X.columns:
    order = np.argsort(X["Open Date_age_days"].values)
    split_idx = int(len(order) * 0.8)
    train_idx, valid_idx = order[:split_idx], order[split_idx:]
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
else:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

model = CatBoostRegressor(
    iterations=4000,
    learning_rate=0.025,
    depth=5,
    l2_leaf_reg=8.0,
    loss_function="RMSE",
    verbose=0,
    random_seed=42,
)

model.fit(X_train, y_train, cat_features=cat_idx, eval_set=(X_valid, y_valid), use_best_model=True)

valid_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

model.fit(X, y, cat_features=cat_idx)
test_pred = model.predict(X_test)

submission = pd.DataFrame({"Id": test["Id"], "Prediction": test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Final Validation Performance: {rmse}")
