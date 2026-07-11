
import os
import sys
import subprocess
import random
import numpy as np
import pandas as pd

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "lightgbm"])

from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
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

for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_cat = X_train.copy()
X_valid_cat = X_valid.copy()
X_test_cat = X_test.copy()

for c in cat_cols:
    X_train_cat[c] = X_train_cat[c].astype(str)
    X_valid_cat[c] = X_valid_cat[c].astype(str)
    X_test_cat[c] = X_test_cat[c].astype(str)

cat_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42,
    verbose=200
)

cat_model.fit(
    X_train_cat,
    y_train,
    cat_features=cat_idx,
    eval_set=(X_valid_cat, y_valid),
    use_best_model=True
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression",
    random_state=42
)

lgb_model.fit(
    X_train,
    y_train,
    categorical_feature=cat_cols
)

valid_pred_cat = cat_model.predict(X_valid_cat)
valid_pred_lgb = lgb_model.predict(X_valid)
valid_pred = 0.5 * valid_pred_cat + 0.5 * valid_pred_lgb
rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))

test_pred_cat = cat_model.predict(X_test_cat)
test_pred_lgb = lgb_model.predict(X_test)
test_pred = 0.5 * test_pred_cat + 0.5 * test_pred_lgb

submission = pd.DataFrame({
    "Id": test["Id"],
    "Prediction": test_pred
})
submission.to_csv("submission_ensemble.csv", index=False)

print(f"Final Validation Performance: {rmse}")
