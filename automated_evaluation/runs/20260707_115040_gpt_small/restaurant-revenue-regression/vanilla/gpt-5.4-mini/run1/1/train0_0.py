
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

cat_cols = ["City", "City Group", "Type"]
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    loss_function="RMSE",
    verbose=0,
    random_seed=42,
)

model.fit(X_train, y_train, cat_features=cat_idx)

valid_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

model.fit(X, y, cat_features=cat_idx)
test_pred = model.predict(X_test)

submission = pd.DataFrame({"Id": test["Id"], "Prediction": test_pred})
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Final Validation Performance: {rmse}")
