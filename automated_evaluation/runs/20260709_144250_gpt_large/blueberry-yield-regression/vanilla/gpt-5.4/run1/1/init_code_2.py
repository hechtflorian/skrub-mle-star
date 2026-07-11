
import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression_l1",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="l1",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
)

valid_pred = model.predict(X_valid)
final_validation_score = mean_absolute_error(y_valid, valid_pred)

best_n_estimators = model.best_iteration_

full_model = lgb.LGBMRegressor(
    n_estimators=best_n_estimators,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression_l1",
    random_state=42
)

full_model.fit(X, y)

test_pred = full_model.predict(X_test)

submission = pd.DataFrame({
    "id": test["id"],
    "yield": test_pred
})
submission.to_csv("submission_lightgbm.csv", index=False)

print(f"Final Validation Performance: {final_validation_score}")
