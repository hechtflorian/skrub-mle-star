
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

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

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1
model1 = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    verbose=0,
    random_seed=42,
)

# Model 2: kept separate with a slightly different seed for diversity
model2 = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    verbose=0,
    random_seed=7,
)

model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

valid_pred1 = model1.predict(X_valid)
valid_pred2 = model2.predict(X_valid)

test_pred1 = model1.predict(X_test)
test_pred2 = model2.predict(X_test)

rmse1 = np.sqrt(mean_squared_error(y_valid, valid_pred1))
rmse2 = np.sqrt(mean_squared_error(y_valid, valid_pred2))

# Decide rank mix: default 50/50, or 60/40 if validation RMSEs differ noticeably
if abs(rmse1 - rmse2) <= 0.05 * min(rmse1, rmse2):
    w1, w2 = 0.5, 0.5
else:
    if rmse1 < rmse2:
        w1, w2 = 0.6, 0.4
    else:
        w1, w2 = 0.4, 0.6

def percentile_rank(arr):
    s = pd.Series(arr)
    return s.rank(method="average", pct=True).values

# Blend on raw predictions
valid_rank_raw_1 = percentile_rank(valid_pred1)
valid_rank_raw_2 = percentile_rank(valid_pred2)
blend_valid_raw = w1 * valid_rank_raw_1 + w2 * valid_rank_raw_2

test_rank_raw_1 = percentile_rank(test_pred1)
test_rank_raw_2 = percentile_rank(test_pred2)
blend_test_raw = w1 * test_rank_raw_1 + w2 * test_rank_raw_2

# Blend on log1p-transformed predictions
valid_log1p_1 = np.log1p(np.maximum(valid_pred1, 0))
valid_log1p_2 = np.log1p(np.maximum(valid_pred2, 0))
blend_valid_log = (
    w1 * percentile_rank(valid_log1p_1) + w2 * percentile_rank(valid_log1p_2)
)

test_log1p_1 = np.log1p(np.maximum(test_pred1, 0))
test_log1p_2 = np.log1p(np.maximum(test_pred2, 0))
blend_test_log = w1 * percentile_rank(test_log1p_1) + w2 * percentile_rank(test_log1p_2)

# Average the two blended ranks
blend_valid = 0.5 * (blend_valid_raw + blend_valid_log)
blend_test = 0.5 * (blend_test_raw + blend_test_log)

# Tiny linear calibration: y ≈ a * blended_rank + b
a, b = np.polyfit(blend_valid, y_valid, 1)
test_pred = a * blend_test + b

submission = pd.DataFrame({"Id": test["Id"], "Prediction": test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)

final_validation_score = np.sqrt(mean_squared_error(y_valid, a * blend_valid + b))
print(f"Final Validation Performance: {final_validation_score}")
