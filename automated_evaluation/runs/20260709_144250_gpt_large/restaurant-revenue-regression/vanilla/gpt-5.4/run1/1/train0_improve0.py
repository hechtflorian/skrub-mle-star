
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

try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
    from sklearn.model_selection import KFold
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

X = train.drop(columns=["revenue"])
y = train["revenue"].copy()
X_test = test.copy()

cat_cols = ["City", "City Group", "Type"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid = [
    {
        "depth": 6,
        "l2_leaf_reg": 3,
        "bagging_temperature": 0.5,
        "random_strength": 1.0,
    },
    {
        "depth": 7,
        "l2_leaf_reg": 5,
        "bagging_temperature": 1.0,
        "random_strength": 1.5,
    },
    {
        "depth": 8,
        "l2_leaf_reg": 7,
        "bagging_temperature": 1.5,
        "random_strength": 2.0,
    },
]

best_params = None
best_cv_rmse = float("inf")

for params in param_grid:
    fold_rmses = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.03,
            depth=params["depth"],
            l2_leaf_reg=params["l2_leaf_reg"],
            bagging_temperature=params["bagging_temperature"],
            random_strength=params["random_strength"],
            loss_function="RMSE",
            eval_metric="RMSE",
            bootstrap_type="Bayesian",
            random_seed=42 + fold,
            verbose=False
        )

        model.fit(
            X_train,
            y_train,
            cat_features=cat_idx,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=200
        )

        valid_pred = model.predict(X_valid)
        fold_rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))
        fold_rmses.append(fold_rmse)

    mean_cv_rmse = float(np.mean(fold_rmses))

    if mean_cv_rmse < best_cv_rmse:
        best_cv_rmse = mean_cv_rmse
        best_params = params

oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))
fold_rmses = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=best_params["depth"],
        l2_leaf_reg=best_params["l2_leaf_reg"],
        bagging_temperature=best_params["bagging_temperature"],
        random_strength=best_params["random_strength"],
        loss_function="RMSE",
        eval_metric="RMSE",
        bootstrap_type="Bayesian",
        random_seed=42 + fold,
        verbose=200
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=200
    )

    valid_pred = model.predict(X_valid)
    oof_pred[valid_idx] = valid_pred
    fold_rmses.append(float(np.sqrt(mean_squared_error(y_valid, valid_pred))))
    test_pred += model.predict(X_test) / kf.n_splits

rmse = float(np.sqrt(mean_squared_error(y, oof_pred)))
cv_rmse_mean = float(np.mean(fold_rmses))
cv_rmse_std = float(np.std(fold_rmses))

submission = pd.DataFrame({
    "Id": test["Id"],
    "Prediction": test_pred
})
submission.to_csv("submission_catboost.csv", index=False)

print(f"Final Validation Performance: {rmse}")
