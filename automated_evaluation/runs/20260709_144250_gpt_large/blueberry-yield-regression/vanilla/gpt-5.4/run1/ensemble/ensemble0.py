
import os
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
lgb_valid_preds = []

for seed in cat_seeds:
    cat_model = CatBoostRegressor(
        iterations=2000,
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
        use_best_model=False
    )
    cat_valid_preds.append(cat_model.predict(X_valid))

for seed in lgb_seeds:
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
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
        callbacks=[lgb.log_evaluation(200)]
    )
    lgb_valid_preds.append(lgb_model.predict(X_valid))

cat_valid_pred = np.mean(cat_valid_preds, axis=0)
lgb_valid_pred = np.mean(lgb_valid_preds, axis=0)

best_score = float("inf")
best_cat_weight = 0.5
best_lgb_weight = 0.5

for cat_weight in np.arange(0.0, 1.0001, 0.05):
    lgb_weight = 1.0 - cat_weight
    blended_valid_pred = cat_weight * cat_valid_pred + lgb_weight * lgb_valid_pred
    score = mean_absolute_error(y_valid, blended_valid_pred)
    if score < best_score:
        best_score = score
        best_cat_weight = cat_weight
        best_lgb_weight = lgb_weight

valid_pred = best_cat_weight * cat_valid_pred + best_lgb_weight * lgb_valid_pred
final_validation_score = mean_absolute_error(y_valid, valid_pred)

full_cat_test_preds = []
full_lgb_test_preds = []

for seed in cat_seeds:
    full_cat_model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=seed,
        verbose=200
    )
    full_cat_model.fit(X, y)
    full_cat_test_preds.append(full_cat_model.predict(X_test))

for seed in lgb_seeds:
    full_lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="regression_l1",
        random_state=seed
    )
    full_lgb_model.fit(X, y)
    full_lgb_test_preds.append(full_lgb_model.predict(X_test))

cat_test_pred = np.mean(full_cat_test_preds, axis=0)
lgb_test_pred = np.mean(full_lgb_test_preds, axis=0)

test_pred = best_cat_weight * cat_test_pred + best_lgb_weight * lgb_test_pred

submission = pd.DataFrame({
    "id": test["id"],
    "yield": test_pred
})
submission.to_csv("submission_ensemble.csv", index=False)

alt_weights = sorted(set([
    round(best_cat_weight, 2),
    round(max(0.0, best_cat_weight - 0.05), 2),
    round(min(1.0, best_cat_weight + 0.05), 2)
]))

for w in alt_weights:
    alt_pred = w * cat_test_pred + (1.0 - w) * lgb_test_pred
    alt_submission = pd.DataFrame({
        "id": test["id"],
        "yield": alt_pred
    })
    alt_submission.to_csv(f"submission_ensemble_cat_{w:.2f}.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
