
import os
import random
import numpy as np
import pandas as pd
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

cat_cols = ["City", "City Group", "Type"]
for c in cat_cols:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")

X = train.drop(columns=["revenue"])
y = train["revenue"].copy()
X_test = test.copy()

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="regression",
    random_state=42
)

model.fit(
    X_train,
    y_train,
    categorical_feature=cat_cols
)

valid_pred = model.predict(X_valid)
rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))

test_pred = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test["Id"],
    "Prediction": test_pred
})
submission.to_csv("submission_lgbm.csv", index=False)

print(f"Final Validation Performance: {rmse}")
