
import os
import sys
import subprocess
import numpy as np
import pandas as pd

def ensure_package(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_package("lightgbm")
ensure_package("catboost")
ensure_package("scikit-learn", "sklearn")

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

train_path = os.path.join(".", "input", "train.csv")
test_path = os.path.join(".", "input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = train.drop(columns=["id", "yield"])
y = train["yield"]
X_test = test.drop(columns=["id"])

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

seed_list = [42, 52, 62]

cat_valid_preds = []
lgb_valid_preds = []

cat_test_preds = []
lgb_test_preds = []

for seed in seed_list:
    cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=200
    )

    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression_l1",
        random_state=seed
    )

    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=False
    )

    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[lgb.log_evaluation(200)]
    )

    cat_valid_preds.append(cat_model.predict(X_valid))
    lgb_valid_preds.append(lgb_model.predict(X_valid))

cat_valid_pred = np.mean(cat_valid_preds, axis=0)
lgb_valid_pred = np.mean(lgb_valid_preds, axis=0)

blend_candidates = [
    (0.60, 0.40),
    (0.65, 0.35),
    (0.70, 0.30),
    (0.75, 0.25)
]

best_raw_score = float("inf")
best_cat_weight_raw, best_lgb_weight_raw = 0.5, 0.5

for cat_weight, lgb_weight in blend_candidates:
    blended_valid_pred = cat_weight * cat_valid_pred + lgb_weight * lgb_valid_pred
    score = mean_absolute_error(y_valid, blended_valid_pred)
    if score < best_raw_score:
        best_raw_score = score
        best_cat_weight_raw, best_lgb_weight_raw = cat_weight, lgb_weight

raw_valid_pred = best_cat_weight_raw * cat_valid_pred + best_lgb_weight_raw * lgb_valid_pred

def to_ranks(values):
    return pd.Series(values).rank(method="average").to_numpy()

def calibrate_by_order_same_length(blended_ranks, target_values):
    blended_ranks = np.asarray(blended_ranks)
    sorted_targets = np.sort(np.asarray(target_values))
    if len(blended_ranks) != len(sorted_targets):
        raise ValueError(
            f"Length mismatch: blended_ranks has length {len(blended_ranks)} "
            f"but target_values has length {len(sorted_targets)}"
        )
    order = np.argsort(blended_ranks, kind="mergesort")
    calibrated = np.empty_like(blended_ranks, dtype=float)
    calibrated[order] = sorted_targets
    return calibrated

def calibrate_by_quantiles(blended_ranks, target_values):
    blended_ranks = np.asarray(blended_ranks, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    n_pred = len(blended_ranks)
    n_target = len(target_values)

    if n_pred == 0:
        return np.array([], dtype=float)

    sorted_targets = np.sort(target_values)
    order = np.argsort(blended_ranks, kind="mergesort")
    calibrated = np.empty(n_pred, dtype=float)

    if n_pred == 1:
        calibrated[order[0]] = np.median(sorted_targets)
        return calibrated

    quantile_positions = np.linspace(0, n_target - 1, n_pred)
    mapped_targets = np.interp(
        quantile_positions,
        np.arange(n_target),
        sorted_targets
    )
    calibrated[order] = mapped_targets
    return calibrated

cat_valid_rank = to_ranks(cat_valid_pred)
lgb_valid_rank = to_ranks(lgb_valid_pred)

rank_weight_grid = np.arange(0.0, 1.0001, 0.05)

best_rank_score = float("inf")
best_cat_weight_rank = 0.5

for cat_weight in rank_weight_grid:
    lgb_weight = 1.0 - cat_weight
    blended_rank_valid = cat_weight * cat_valid_rank + lgb_weight * lgb_valid_rank
    calibrated_rank_valid_pred = calibrate_by_order_same_length(blended_rank_valid, y_valid.values)
    score = mean_absolute_error(y_valid, calibrated_rank_valid_pred)
    if score < best_rank_score:
        best_rank_score = score
        best_cat_weight_rank = cat_weight

best_lgb_weight_rank = 1.0 - best_cat_weight_rank
best_rank_valid = best_cat_weight_rank * cat_valid_rank + best_lgb_weight_rank * lgb_valid_rank
rank_valid_pred = calibrate_by_order_same_length(best_rank_valid, y_valid.values)

alpha_grid = [0.25, 0.50, 0.75]
best_hybrid_score = float("inf")
best_alpha = 0.5

for alpha in alpha_grid:
    hybrid_valid_pred = alpha * raw_valid_pred + (1.0 - alpha) * rank_valid_pred
    score = mean_absolute_error(y_valid, hybrid_valid_pred)
    if score < best_hybrid_score:
        best_hybrid_score = score
        best_alpha = alpha

valid_pred = best_alpha * raw_valid_pred + (1.0 - best_alpha) * rank_valid_pred
final_validation_score = mean_absolute_error(y_valid, valid_pred)

for seed in seed_list:
    full_cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=200
    )

    full_lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression_l1",
        random_state=seed
    )

    full_cat_model.fit(X, y)
    full_lgb_model.fit(X, y)

    cat_test_preds.append(full_cat_model.predict(X_test))
    lgb_test_preds.append(full_lgb_model.predict(X_test))

cat_test_pred = np.mean(cat_test_preds, axis=0)
lgb_test_pred = np.mean(lgb_test_preds, axis=0)

raw_test_pred = best_cat_weight_raw * cat_test_pred + best_lgb_weight_raw * lgb_test_pred

cat_test_rank = to_ranks(cat_test_pred)
lgb_test_rank = to_ranks(lgb_test_pred)
blended_test_rank = best_cat_weight_rank * cat_test_rank + best_lgb_weight_rank * lgb_test_rank

rank_test_pred = calibrate_by_quantiles(blended_test_rank, y.values)

test_pred = best_alpha * raw_test_pred + (1.0 - best_alpha) * rank_test_pred

submission = pd.DataFrame({
    "id": test["id"],
    "yield": test_pred
})
submission.to_csv("submission_ensemble.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
