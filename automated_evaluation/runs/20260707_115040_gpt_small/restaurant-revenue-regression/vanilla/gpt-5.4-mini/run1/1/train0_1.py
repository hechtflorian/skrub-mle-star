
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

df = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Feature engineering for date
for c in ["Open Date"]:
    df[c] = pd.to_datetime(df[c])
    test[c] = pd.to_datetime(test[c])
    for f in ["year", "month", "day"]:
        df[f"{c}_{f}"] = getattr(df[c].dt, f)
        test[f"{c}_{f}"] = getattr(test[c].dt, f)

df = df.drop(columns=["Open Date"])
test = test.drop(columns=["Open Date"])

y = df["revenue"].values
X = df.drop(columns=["revenue"])
X_test = test.copy()

# CatBoost categorical features
cat_cols = ["City", "City Group", "Type"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

# LightGBM one-hot encoded features
X_lgb = pd.get_dummies(X, columns=cat_cols)
X_test_lgb = pd.get_dummies(X_test, columns=cat_cols)
X_lgb, X_test_lgb = X_lgb.align(X_test_lgb, join="left", axis=1, fill_value=0)

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train_lgb, X_valid_lgb, _, _ = train_test_split(
    X_lgb, y, test_size=0.2, random_state=42
)

# CatBoost model
cat_model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    verbose=0,
    random_seed=42,
)

cat_model.fit(X_train, y_train, cat_features=cat_idx)
cat_valid_pred = cat_model.predict(X_valid)

# LightGBM model
lgb_model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

lgb_model.fit(X_train_lgb, y_train)
lgb_valid_pred = lgb_model.predict(X_valid_lgb)

# Ensemble validation predictions
valid_pred = 0.5 * cat_valid_pred + 0.5 * lgb_valid_pred
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

# Fit on full data
cat_model.fit(X, y, cat_features=cat_idx)
lgb_model.fit(X_lgb, y)

# Predict test set
cat_test_pred = cat_model.predict(X_test)
lgb_test_pred = lgb_model.predict(X_test_lgb)
test_pred = 0.5 * cat_test_pred + 0.5 * lgb_test_pred

submission = pd.DataFrame({"Id": test["Id"], "Prediction": test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Final Validation Performance: {rmse}")
