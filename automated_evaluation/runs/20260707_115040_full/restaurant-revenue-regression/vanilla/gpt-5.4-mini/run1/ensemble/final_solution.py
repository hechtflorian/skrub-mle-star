
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
FINAL_DIR = "./final"
SUBMISSION_PATH = os.path.join(FINAL_DIR, "submission.csv")

os.makedirs(FINAL_DIR, exist_ok=True)

df = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Remove Open Date-derived features and keep the pipeline lightweight
df = df.drop(columns=["Open Date"])
test = test.drop(columns=["Open Date"])

# Simple numeric encoding for categorical columns
cat_cols = ["City", "City Group", "Type"]
for c in cat_cols:
    combined = pd.concat([df[c], test[c]], axis=0)
    codes, uniques = pd.factorize(combined, sort=True)
    df[c] = codes[: len(df)].astype("int32")
    test[c] = codes[len(df) :].astype("int32")

y = df["revenue"].values.astype(float)
X = df.drop(columns=["revenue"])
X_test = test.copy()

# Use the full training set, as requested
model1 = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    verbose=0,
    random_seed=42,
)

model2 = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    verbose=0,
    random_seed=7,
)

model1.fit(X, y)
model2.fit(X, y)

test_pred1 = model1.predict(X_test)
test_pred2 = model2.predict(X_test)

def percentile_rank(arr):
    s = pd.Series(arr)
    return s.rank(method="average", pct=True).values

test_rank_raw_1 = percentile_rank(test_pred1)
test_rank_raw_2 = percentile_rank(test_pred2)
blend_test_raw = 0.5 * test_rank_raw_1 + 0.5 * test_rank_raw_2

test_log1p_1 = np.log1p(np.maximum(test_pred1, 0))
test_log1p_2 = np.log1p(np.maximum(test_pred2, 0))
blend_test_log = 0.5 * percentile_rank(test_log1p_1) + 0.5 * percentile_rank(test_log1p_2)

blend_test = 0.5 * (blend_test_raw + blend_test_log)

# Calibrate on the full training set using an in-sample approximation
train_pred1 = model1.predict(X)
train_pred2 = model2.predict(X)
train_rank_raw_1 = percentile_rank(train_pred1)
train_rank_raw_2 = percentile_rank(train_pred2)
blend_train_raw = 0.5 * train_rank_raw_1 + 0.5 * train_rank_raw_2

train_log1p_1 = np.log1p(np.maximum(train_pred1, 0))
train_log1p_2 = np.log1p(np.maximum(train_pred2, 0))
blend_train_log = 0.5 * percentile_rank(train_log1p_1) + 0.5 * percentile_rank(train_log1p_2)

blend_train = 0.5 * (blend_train_raw + blend_train_log)

a, b = np.polyfit(blend_train, y, 1)
test_pred = a * blend_test + b

submission = pd.DataFrame({"Id": test["Id"], "Prediction": test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)
