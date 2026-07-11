
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

submission = pd.DataFrame({"id": test["id"], "Rings": test_pred})
submission.to_csv("submission.csv", index=False)
