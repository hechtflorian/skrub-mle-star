
import os
import sys
import subprocess
import random
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

random.seed(42)
np.random.seed(42)

train_path = os.path.join(".", "input", "train.csv")
test_path = os.path.join(".", "input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

for df in [train, test]:
    df["Open Date"] = pd.to_datetime(df["Open Date"], format="%m/%d/%Y")
    df["open_year"] = df["Open Date"].dt.year
    df["open_month"] = df["Open Date"].dt.month
    df["open_day"] = df["Open Date"].dt.day
    df["open_weekday"] = df["Open Date"].dt.weekday
    reference_date = pd.Timestamp("2015-01-01")
    df["restaurant_age_days"] = (reference_date - df["Open Date"]).dt.days
    df.drop(columns=["Open Date"], inplace=True)

X = train.drop(columns=["revenue"]).copy()
y = train["revenue"].copy()
X_test = test.copy()

cat_cols = ["City", "City Group", "Type"]

p_cols = [c for c in X.columns if str(c).startswith("P")]

if len(p_cols) > 0:
    train_zero_ratio = (X[p_cols] == 0).mean(axis=0)
    sparse_p_cols = train_zero_ratio[train_zero_ratio >= 0.5].index.tolist()
    dense_p_cols = [c for c in p_cols if c not in sparse_p_cols]

    for df in (X, X_test):
        df["P_sum"] = df[p_cols].sum(axis=1)
        df["P_mean"] = df[p_cols].mean(axis=1)
        df["P_std"] = df[p_cols].std(axis=1).fillna(0)
        df["P_zero_count"] = (df[p_cols] == 0).sum(axis=1)
        df["P_zero_ratio"] = (df[p_cols] == 0).mean(axis=1)

        if len(sparse_p_cols) > 0:
            df["P_sparse_sum"] = df[sparse_p_cols].sum(axis=1)
            df["P_sparse_mean"] = df[sparse_p_cols].mean(axis=1)
            df["P_sparse_std"] = df[sparse_p_cols].std(axis=1).fillna(0)
            df["P_sparse_zero_count"] = (df[sparse_p_cols] == 0).sum(axis=1)
        else:
            df["P_sparse_sum"] = 0.0
            df["P_sparse_mean"] = 0.0
            df["P_sparse_std"] = 0.0
            df["P_sparse_zero_count"] = 0

        if len(dense_p_cols) > 0:
            df["P_dense_sum"] = df[dense_p_cols].sum(axis=1)
            df["P_dense_mean"] = df[dense_p_cols].mean(axis=1)
            df["P_dense_std"] = df[dense_p_cols].std(axis=1).fillna(0)
            df["P_dense_zero_count"] = (df[dense_p_cols] == 0).sum(axis=1)
        else:
            df["P_dense_sum"] = 0.0
            df["P_dense_mean"] = 0.0
            df["P_dense_std"] = 0.0
            df["P_dense_zero_count"] = 0

        df["P_sum_per_nonzero"] = df["P_sum"] / (len(p_cols) - df["P_zero_count"] + 1)
        df["P_sparse_dense_sum_ratio"] = df["P_sparse_sum"] / (df["P_dense_sum"] + 1.0)

cat_idx = [X.columns.get_loc(c) for c in cat_cols]

y_log = np.log1p(y)

seed_list = [42, 52, 62, 72, 82]
rmse_per_seed = []
test_pred_log_folds = []

for seed in seed_list:
    X_train, X_valid, y_train_log, y_valid_log, y_train_raw, y_valid_raw = train_test_split(
        X, y_log, y, test_size=0.2, random_state=seed
    )

    model = CatBoostRegressor(
        iterations=1800,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=8,
        random_strength=1.5,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=seed,
        verbose=200
    )

    model.fit(
        X_train,
        y_train_log,
        cat_features=cat_idx,
        eval_set=(X_valid, y_valid_log),
        use_best_model=True
    )

    valid_pred_log = model.predict(X_valid)
    valid_pred = np.expm1(valid_pred_log)
    valid_pred = np.clip(valid_pred, 0, None)

    rmse = float(np.sqrt(mean_squared_error(y_valid_raw, valid_pred)))
    rmse_per_seed.append(rmse)

    test_pred_log_folds.append(model.predict(X_test))

final_validation_score = float(np.mean(rmse_per_seed))

test_pred = np.expm1(np.mean(test_pred_log_folds, axis=0))
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "Id": test["Id"],
    "Prediction": test_pred
})
submission.to_csv("submission_catboost.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
