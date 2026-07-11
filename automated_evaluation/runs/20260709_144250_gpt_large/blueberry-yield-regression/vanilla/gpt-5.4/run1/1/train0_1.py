
import os
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

cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=200
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression_l1",
    random_state=42
)

cat_model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
)

cat_valid_pred = cat_model.predict(X_valid)
lgb_valid_pred = lgb_model.predict(X_valid)
valid_pred = 0.5 * cat_valid_pred + 0.5 * lgb_valid_pred
final_validation_score = mean_absolute_error(y_valid, valid_pred)

cat_best_iterations = cat_model.get_best_iteration() + 1
lgb_best_n_estimators = lgb_model.best_iteration_

full_cat_model = CatBoostRegressor(
    iterations=cat_best_iterations,
    learning_rate=0.03,
    depth=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=42,
    verbose=200
)

full_lgb_model = lgb.LGBMRegressor(
    n_estimators=lgb_best_n_estimators,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression_l1",
    random_state=42
)

full_cat_model.fit(X, y)
full_lgb_model.fit(X, y)

cat_test_pred = full_cat_model.predict(X_test)
lgb_test_pred = full_lgb_model.predict(X_test)
test_pred = 0.5 * cat_test_pred + 0.5 * lgb_test_pred

submission = pd.DataFrame({
    "id": test["id"],
    "yield": test_pred
})
submission.to_csv("submission_ensemble.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
