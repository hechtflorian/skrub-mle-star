
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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

    df[c + "_year"] = df[c].dt.year
    df[c + "_month"] = df[c].dt.month
    df[c + "_day"] = df[c].dt.day

    test[c + "_year"] = test[c].dt.year
    test[c + "_month"] = test[c].dt.month
    test[c + "_day"] = test[c].dt.day

df = df.drop(columns=["Open Date"])
test = test.drop(columns=["Open Date"])

y = df["revenue"].values
X = df.drop(columns=["revenue"])
X_test = test.copy()

# One-hot encode categorical columns
X = pd.get_dummies(X, columns=["City", "City Group", "Type"])
X_test = pd.get_dummies(X_test, columns=["City", "City Group", "Type"])

# Align train/test columns
X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

# Train/validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)
rmse = np.sqrt(mean_squared_error(y_valid, valid_pred))

# Fit on full data and predict test set
model.fit(X, y)
test_pred = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test["Id"],
    "Prediction": test_pred
})
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"Final Validation Performance: {rmse}")
