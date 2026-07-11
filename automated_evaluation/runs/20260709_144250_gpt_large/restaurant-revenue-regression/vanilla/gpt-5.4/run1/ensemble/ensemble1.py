
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

best_params_model_1 = None
best_cv_rmse_model_1 = float("inf")

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

    if mean_cv_rmse < best_cv_rmse_model_1:
        best_cv_rmse_model_1 = mean_cv_rmse
        best_params_model_1 = params

oof_pred_1 = np.zeros(len(X))
test_pred_1 = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=best_params_model_1["depth"],
        l2_leaf_reg=best_params_model_1["l2_leaf_reg"],
        bagging_temperature=best_params_model_1["bagging_temperature"],
        random_strength=best_params_model_1["random_strength"],
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
    oof_pred_1[valid_idx] = valid_pred
    test_pred_1 += model.predict(X_test) / kf.n_splits

# Second pipeline: keep the same overall structure, but model log1p(target) and back-transform.
best_params_model_2 = None
best_cv_rmse_model_2 = float("inf")
y_log = np.log1p(y)

for params in param_grid:
    fold_rmses = []

    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train_log, y_valid = y_log.iloc[train_idx], y.iloc[valid_idx]

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
            y_train_log,
            cat_features=cat_idx,
            eval_set=(X_valid, np.log1p(y_valid)),
            use_best_model=True,
            early_stopping_rounds=200
        )

        valid_pred = np.expm1(model.predict(X_valid))
        valid_pred = np.clip(valid_pred, 0, None)
        fold_rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))
        fold_rmses.append(fold_rmse)

    mean_cv_rmse = float(np.mean(fold_rmses))

    if mean_cv_rmse < best_cv_rmse_model_2:
        best_cv_rmse_model_2 = mean_cv_rmse
        best_params_model_2 = params

oof_pred_2 = np.zeros(len(X))
test_pred_2 = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train_log, y_valid = y_log.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=best_params_model_2["depth"],
        l2_leaf_reg=best_params_model_2["l2_leaf_reg"],
        bagging_temperature=best_params_model_2["bagging_temperature"],
        random_strength=best_params_model_2["random_strength"],
        loss_function="RMSE",
        eval_metric="RMSE",
        bootstrap_type="Bayesian",
        random_seed=142 + fold,
        verbose=200
    )

    model.fit(
        X_train,
        y_train_log,
        cat_features=cat_idx,
        eval_set=(X_valid, np.log1p(y_valid)),
        use_best_model=True,
        early_stopping_rounds=200
    )

    valid_pred = np.expm1(model.predict(X_valid))
    valid_pred = np.clip(valid_pred, 0, None)
    oof_pred_2[valid_idx] = valid_pred

    fold_test_pred = np.expm1(model.predict(X_test))
    fold_test_pred = np.clip(fold_test_pred, 0, None)
    test_pred_2 += fold_test_pred / kf.n_splits

def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def make_quantile_edges(values, n_bins):
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.asarray(edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges) - 1):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges

def assign_bins(values, edges):
    bins = np.digitize(values, edges[1:-1], right=False)
    return bins

def piecewise_blend(y_true, oof1, oof2, test1, test2, n_bins):
    avg_oof = 0.5 * (oof1 + oof2)
    avg_test = 0.5 * (test1 + test2)

    edges = make_quantile_edges(avg_oof, n_bins)
    train_bins = assign_bins(avg_oof, edges)
    test_bins = assign_bins(avg_test, edges)

    candidate_weights = np.round(np.arange(0.0, 1.01, 0.1), 1)
    selected_weights = {}

    blended_oof = np.zeros_like(oof1, dtype=float)
    blended_test = np.zeros_like(test1, dtype=float)

    for b in range(n_bins):
        train_mask = train_bins == b
        test_mask = test_bins == b

        if train_mask.sum() == 0:
            w_best = 0.5
        else:
            best_bin_rmse = float("inf")
            w_best = 0.5
            for w in candidate_weights:
                pred_bin = w * oof1[train_mask] + (1.0 - w) * oof2[train_mask]
                score = rmse_score(y_true[train_mask], pred_bin)
                if score < best_bin_rmse:
                    best_bin_rmse = score
                    w_best = float(w)

        w_final = 0.8 * w_best + 0.2 * 0.5
        selected_weights[b] = w_final

        blended_oof[train_mask] = w_final * oof1[train_mask] + (1.0 - w_final) * oof2[train_mask]
        blended_test[test_mask] = w_final * test1[test_mask] + (1.0 - w_final) * test2[test_mask]

    blended_oof = np.clip(blended_oof, 0, None)
    blended_test = np.clip(blended_test, 0, None)
    score = rmse_score(y_true, blended_oof)

    return score, blended_oof, blended_test, edges, selected_weights

score_2bins, blended_oof_2bins, blended_test_2bins, edges_2bins, weights_2bins = piecewise_blend(
    y.values, oof_pred_1, oof_pred_2, test_pred_1, test_pred_2, n_bins=2
)

score_3bins, blended_oof_3bins, blended_test_3bins, edges_3bins, weights_3bins = piecewise_blend(
    y.values, oof_pred_1, oof_pred_2, test_pred_1, test_pred_2, n_bins=3
)

global_best_score = float("inf")
global_best_w = 0.5
global_best_oof = None
global_best_test = None
for w in np.round(np.arange(0.0, 1.01, 0.1), 1):
    oof_blend = w * oof_pred_1 + (1.0 - w) * oof_pred_2
    oof_blend = np.clip(oof_blend, 0, None)
    score = rmse_score(y.values, oof_blend)
    if score < global_best_score:
        global_best_score = score
        global_best_w = float(w)
        global_best_oof = oof_blend
        global_best_test = np.clip(w * test_pred_1 + (1.0 - w) * test_pred_2, 0, None)

if score_3bins <= score_2bins and score_3bins <= global_best_score:
    final_validation_score = score_3bins
    final_test_pred = blended_test_3bins
elif score_2bins <= global_best_score:
    final_validation_score = score_2bins
    final_test_pred = blended_test_2bins
else:
    final_validation_score = global_best_score
    final_test_pred = global_best_test

submission = pd.DataFrame({
    "Id": test["Id"],
    "Prediction": final_test_pred
})
submission.to_csv("submission_catboost.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
