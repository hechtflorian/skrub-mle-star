
import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Preprocessing
X = pd.get_dummies(train.drop(columns=["Rings"]))
y = np.log1p(train["Rings"].astype(float))
X_test = pd.get_dummies(test)

X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# More stable out-of-fold scheme on the log-target
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

oof_lgb_log = np.zeros(len(X))
oof_cat_log = np.zeros(len(X))
test_lgb_log = np.zeros(len(X_test))
test_cat_log = np.zeros(len(X_test))

lgb_best_iterations = []

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # LightGBM model
    lgb_model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold
    )

    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    best_iter = lgb_model.best_iteration_ if lgb_model.best_iteration_ is not None else lgb_model.n_estimators
    lgb_best_iterations.append(best_iter)

    oof_lgb_log[valid_idx] = lgb_model.predict(X_valid, num_iteration=best_iter)
    test_lgb_log += lgb_model.predict(X_test, num_iteration=best_iter) / n_splits

    # Lightweight CatBoost companion model
    cat_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        verbose=False,
        random_seed=SEED + fold
    )

    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    oof_cat_log[valid_idx] = cat_model.predict(X_valid)
    test_cat_log += cat_model.predict(X_test) / n_splits

valid_true = np.expm1(y).clip(0, None)

# Blend on the log scale for a more consistent target transformation
oof_lgb = np.expm1(oof_lgb_log).clip(0, None)
oof_cat = np.expm1(oof_cat_log).clip(0, None)
oof_pred = 0.8 * oof_lgb + 0.2 * oof_cat
final_validation_score = np.sqrt(mean_squared_log_error(valid_true, oof_pred))
print(f"Final Validation Performance: {final_validation_score}")

# Train final models on all data with averaged best iteration for LightGBM
avg_best_iter = int(np.mean(lgb_best_iterations)) if len(lgb_best_iterations) > 0 else 5000

lgb_final_model = lgb.LGBMRegressor(
    n_estimators=avg_best_iter,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)
lgb_final_model.fit(X, y)

cat_final_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    verbose=False,
    random_seed=SEED
)
cat_final_model.fit(X, y)

# Test predictions and ensemble
lgb_test_pred_log = lgb_final_model.predict(X_test)
cat_test_pred_log = cat_final_model.predict(X_test)

lgb_test_pred = np.expm1(lgb_test_pred_log).clip(0, None)
cat_test_pred = np.expm1(cat_test_pred_log).clip(0, None)

test_pred = 0.8 * lgb_test_pred + 0.2 * cat_test_pred
test_pred = np.clip(test_pred, 0, None)

# =========================
# Residual-style ensemble refinement
# Solution 1 = baseline, Solution 2 = residual correction
# =========================

# OOF predictions for both solutions on original scale
sol1_oof = oof_lgb
sol2_oof = oof_cat

# Original scale test predictions for both solutions
sol1_test = lgb_test_pred
sol2_test = cat_test_pred

# Fold-wise stability weighting: inverse fold RMSLE
fold_weights = np.zeros(len(X), dtype=float)
for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    y_true_fold = valid_true[valid_idx]
    sol1_fold = sol1_oof[valid_idx]
    sol2_fold = sol2_oof[valid_idx]

    rmsle1 = np.sqrt(mean_squared_log_error(y_true_fold, np.clip(sol1_fold, 0, None)))
    rmsle2 = np.sqrt(mean_squared_log_error(y_true_fold, np.clip(sol2_fold, 0, None)))

    # If Solution 2 is worse, shrink its effect more on that fold
    w = 1.0 / (rmsle2 + 1e-8)
    fold_weights[valid_idx] = w

# Weighted residual delta for OOF alpha search
delta_oof = sol2_oof - sol1_oof

# Search alpha on a log-like scale for OOF robustness
alphas = np.linspace(-0.5, 1.5, 201)
best_alpha = 0.0
best_score = np.inf

for alpha in alphas:
    blended_oof = sol1_oof + alpha * (fold_weights * delta_oof)
    blended_oof = np.clip(blended_oof, 0, None)
    score = np.sqrt(mean_squared_log_error(valid_true, blended_oof))
    if score < best_score:
        best_score = score
        best_alpha = alpha

# Apply the same alpha on test predictions
final_test = sol1_test + best_alpha * (sol2_test - sol1_test)
final_test = np.clip(final_test, 0, None)

submission = pd.DataFrame({"id": test["id"], "Rings": final_test})
submission.to_csv("submission.csv", index=False)
