
import os
import subprocess
import sys

def ensure_package(package_name, import_name=None):
    import_name = import_name or package_name
    try:
        __import__(import_name)
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

ensure_package("numpy", "numpy")
ensure_package("pandas", "pandas")
ensure_package("lightgbm", "lightgbm")
ensure_package("scikit-learn", "sklearn")
ensure_package("catboost", "catboost")

import numpy as np
import pandas as pd
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

cat_seeds = [42, 52, 62]
lgb_seeds = [42, 52, 62]

cat_valid_preds = []
cat_best_iterations_list = []

for seed in cat_seeds:
    cat_model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=200
    )
    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )
    cat_valid_preds.append(cat_model.predict(X_valid))
    best_iter = cat_model.get_best_iteration()
    if best_iter is None or best_iter < 0:
        best_iter = 5000
    else:
        best_iter = best_iter + 1
    cat_best_iterations_list.append(best_iter)

lgb_valid_preds = []
lgb_best_iterations_list = []

for seed in lgb_seeds:
    lgb_model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression_l1",
        random_state=seed
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    lgb_valid_preds.append(lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration_))
    best_iter = lgb_model.best_iteration_
    if best_iter is None or best_iter <= 0:
        best_iter = 5000
    lgb_best_iterations_list.append(best_iter)

cat_valid_pred = np.mean(cat_valid_preds, axis=0)
lgb_valid_pred = np.mean(lgb_valid_preds, axis=0)

best_weight = None
best_score = float("inf")

for cat_weight in np.arange(0.0, 1.0001, 0.05):
    valid_pred = cat_weight * cat_valid_pred + (1.0 - cat_weight) * lgb_valid_pred
    score = mean_absolute_error(y_valid, valid_pred)
    if score < best_score:
        best_score = score
        best_weight = cat_weight

final_validation_score = best_score

cat_best_iterations = int(round(np.mean(cat_best_iterations_list)))
lgb_best_n_estimators = int(round(np.mean(lgb_best_iterations_list)))

full_cat_models = []
for seed in cat_seeds:
    full_cat_model = CatBoostRegressor(
        iterations=cat_best_iterations,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=200
    )
    full_cat_model.fit(X, y)
    full_cat_models.append(full_cat_model)

full_lgb_models = []
for seed in lgb_seeds:
    full_lgb_model = lgb.LGBMRegressor(
        n_estimators=lgb_best_n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression_l1",
        random_state=seed
    )
    full_lgb_model.fit(X, y)
    full_lgb_models.append(full_lgb_model)

cat_test_pred = np.mean([model.predict(X_test) for model in full_cat_models], axis=0)
lgb_test_pred = np.mean([model.predict(X_test) for model in full_lgb_models], axis=0)
test_pred = best_weight * cat_test_pred + (1.0 - best_weight) * lgb_test_pred

submission = pd.DataFrame({
    "id": test["id"],
    "yield": test_pred
})
submission.to_csv("submission_ensemble.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
