
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


def create_base_features(df):
    df = df.copy()
    df["Open Date"] = pd.to_datetime(df["Open Date"], format="%m/%d/%Y")
    df["open_year"] = df["Open Date"].dt.year
    df["open_month"] = df["Open Date"].dt.month
    df["open_day"] = df["Open Date"].dt.day
    df["open_weekday"] = df["Open Date"].dt.weekday
    reference_date = pd.Timestamp("2015-01-01")
    df["restaurant_age_days"] = (reference_date - df["Open Date"]).dt.days
    df.drop(columns=["Open Date"], inplace=True)
    return df


def create_extra_features(df):
    df = create_base_features(df)
    p_cols = [c for c in df.columns if c.startswith("P")]
    if len(p_cols) > 0:
        df["P_sum"] = df[p_cols].sum(axis=1)
        df["P_mean"] = df[p_cols].mean(axis=1)
        df["P_std"] = df[p_cols].std(axis=1).fillna(0)
        df["P_max"] = df[p_cols].max(axis=1)
        df["P_min"] = df[p_cols].min(axis=1)
        df["P_range"] = df["P_max"] - df["P_min"]
        df["P_nonzero"] = (df[p_cols] != 0).sum(axis=1)
    df["age_years"] = df["restaurant_age_days"] / 365.25
    df["is_new_restaurant"] = (df["restaurant_age_days"] < 365 * 5).astype(int)
    return df


train_base = create_base_features(train)
test_base = create_base_features(test)

train_extra = create_extra_features(train)
test_extra = create_extra_features(test)

X1 = train_base.drop(columns=["revenue"])
y = train_base["revenue"].copy()
X1_test = test_base.copy()

cat_cols_1 = ["City", "City Group", "Type"]
cat_idx_1 = [X1.columns.get_loc(c) for c in cat_cols_1]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

param_grid_1 = [
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

best_params_1 = None
best_cv_rmse_1 = float("inf")

for params in param_grid_1:
    fold_rmses = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X1, y), 1):
        X_train, X_valid = X1.iloc[train_idx], X1.iloc[valid_idx]
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
            cat_features=cat_idx_1,
            eval_set=(X_valid, y_valid),
            use_best_model=True,
            early_stopping_rounds=200
        )

        valid_pred = model.predict(X_valid)
        fold_rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))
        fold_rmses.append(fold_rmse)

    mean_cv_rmse = float(np.mean(fold_rmses))
    if mean_cv_rmse < best_cv_rmse_1:
        best_cv_rmse_1 = mean_cv_rmse
        best_params_1 = params

oof_pred_1 = np.zeros(len(X1))
test_pred_1 = np.zeros(len(X1_test))
fold_rmses_1 = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X1, y), 1):
    X_train, X_valid = X1.iloc[train_idx], X1.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=best_params_1["depth"],
        l2_leaf_reg=best_params_1["l2_leaf_reg"],
        bagging_temperature=best_params_1["bagging_temperature"],
        random_strength=best_params_1["random_strength"],
        loss_function="RMSE",
        eval_metric="RMSE",
        bootstrap_type="Bayesian",
        random_seed=42 + fold,
        verbose=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx_1,
        eval_set=(X_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=200
    )

    valid_pred = model.predict(X_valid)
    oof_pred_1[valid_idx] = valid_pred
    fold_rmses_1.append(float(np.sqrt(mean_squared_error(y_valid, valid_pred))))
    test_pred_1 += model.predict(X1_test) / kf.n_splits

submission_1 = pd.DataFrame({
    "Id": test["Id"],
    "Prediction": test_pred_1
})
submission_1.to_csv("submission_catboost_raw.csv", index=False)

X2 = train_extra.drop(columns=["revenue"])
y2 = np.log1p(train_extra["revenue"].copy())
X2_test = test_extra.copy()

cat_cols_2 = ["City", "City Group", "Type"]
cat_idx_2 = [X2.columns.get_loc(c) for c in cat_cols_2]

param_grid_2 = [
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

best_params_2 = None
best_cv_rmse_2 = float("inf")

for params in param_grid_2:
    fold_rmses = []
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X2, y2), 1):
        X_train, X_valid = X2.iloc[train_idx], X2.iloc[valid_idx]
        y_train, y_valid_log = y2.iloc[train_idx], y2.iloc[valid_idx]
        y_valid_raw = np.expm1(y_valid_log)

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
            random_seed=142 + fold,
            verbose=False
        )

        model.fit(
            X_train,
            y_train,
            cat_features=cat_idx_2,
            eval_set=(X_valid, y_valid_log),
            use_best_model=True,
            early_stopping_rounds=200
        )

        valid_pred_log = model.predict(X_valid)
        valid_pred = np.expm1(valid_pred_log)
        valid_pred = np.clip(valid_pred, 0, None)
        fold_rmse = float(np.sqrt(mean_squared_error(y_valid_raw, valid_pred)))
        fold_rmses.append(fold_rmse)

    mean_cv_rmse = float(np.mean(fold_rmses))
    if mean_cv_rmse < best_cv_rmse_2:
        best_cv_rmse_2 = mean_cv_rmse
        best_params_2 = params

oof_pred_2 = np.zeros(len(X2))
test_pred_2 = np.zeros(len(X2_test))
fold_rmses_2 = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X2, y2), 1):
    X_train, X_valid = X2.iloc[train_idx], X2.iloc[valid_idx]
    y_train, y_valid_log = y2.iloc[train_idx], y2.iloc[valid_idx]
    y_valid_raw = np.expm1(y_valid_log)

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=best_params_2["depth"],
        l2_leaf_reg=best_params_2["l2_leaf_reg"],
        bagging_temperature=best_params_2["bagging_temperature"],
        random_strength=best_params_2["random_strength"],
        loss_function="RMSE",
        eval_metric="RMSE",
        bootstrap_type="Bayesian",
        random_seed=142 + fold,
        verbose=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_idx_2,
        eval_set=(X_valid, y_valid_log),
        use_best_model=True,
        early_stopping_rounds=200
    )

    valid_pred_log = model.predict(X_valid)
    valid_pred = np.expm1(valid_pred_log)
    valid_pred = np.clip(valid_pred, 0, None)
    oof_pred_2[valid_idx] = valid_pred
    fold_rmses_2.append(float(np.sqrt(mean_squared_error(y_valid_raw, valid_pred))))

    fold_test_pred_log = model.predict(X2_test)
    fold_test_pred = np.expm1(fold_test_pred_log)
    fold_test_pred = np.clip(fold_test_pred, 0, None)
    test_pred_2 += fold_test_pred / kf.n_splits

submission_2 = pd.DataFrame({
    "Id": test["Id"],
    "Prediction": test_pred_2
})
submission_2.to_csv("submission_catboost_logfeat.csv", index=False)

weights = [i / 10.0 for i in range(11)]
best_weight = 0.6
best_blend_rmse = float("inf")

for w in weights:
    blend_oof = w * oof_pred_1 + (1.0 - w) * oof_pred_2
    blend_oof = np.clip(blend_oof, 0, None)
    rmse = float(np.sqrt(mean_squared_error(y, blend_oof)))
    if rmse < best_blend_rmse:
        best_blend_rmse = rmse
        best_weight = w

final_test_pred = best_weight * test_pred_1 + (1.0 - best_weight) * test_pred_2
final_test_pred = np.clip(final_test_pred, 0, None)

submission_ensemble = pd.DataFrame({
    "Id": test["Id"],
    "Prediction": final_test_pred
})
submission_ensemble.to_csv("submission_ensemble.csv", index=False)

print(f"Final Validation Performance: {best_blend_rmse}")
