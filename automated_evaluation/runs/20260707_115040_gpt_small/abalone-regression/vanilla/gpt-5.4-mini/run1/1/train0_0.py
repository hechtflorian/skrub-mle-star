
import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = pd.get_dummies(train.drop(columns=["Rings"]))
y = np.log1p(train["Rings"].astype(float))
X_test = pd.get_dummies(test)

X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)

model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

valid_pred_log = model.predict(X_valid, num_iteration=model.best_iteration_)
valid_pred = np.expm1(valid_pred_log).clip(0, None)
valid_true = np.expm1(y_valid).clip(0, None)

final_validation_score = np.sqrt(mean_squared_log_error(valid_true, valid_pred))
print(f"Final Validation Performance: {final_validation_score}")

final_model = lgb.LGBMRegressor(
    n_estimators=model.best_iteration_ if model.best_iteration_ is not None else 5000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)
final_model.fit(X, y)

test_pred_log = final_model.predict(X_test)
test_pred = np.expm1(test_pred_log).clip(0, None)

submission = pd.DataFrame({"id": test["id"], "Rings": test_pred})
submission.to_csv("submission.csv", index=False)
