
import os
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"])
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["weekday"] = dt.dt.weekday
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    return df

train = add_datetime_features(train)
test = add_datetime_features(test)

features = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "day", "month", "year", "weekday",
    "dayofyear", "weekofyear"
]

X = train[features]
y = train["count"].astype(float)

# Hold-out validation: last 20% by time
train = train.sort_values("datetime").reset_index(drop=True)
split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx].copy()
valid_part = train.iloc[split_idx:].copy()

X_train = train_part[features]
y_train = train_part["count"].astype(float)
X_valid = valid_part[features]
y_valid = valid_part["count"].astype(float)

model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)
valid_pred = np.clip(valid_pred, 0, None)
valid_pred = np.round(valid_pred, 6)

# Reasonable metric for this task: RMSLE
final_validation_score = np.sqrt(mean_squared_log_error(y_valid, np.clip(valid_pred, 0, None)))
print(f"Final Validation Performance: {final_validation_score}")

# Train final model on all training data and predict test
model.fit(X, y)
test_pred = model.predict(test[features])
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "datetime": test["datetime"],
    "count": test_pred
})

submission.to_csv("submission.csv", index=False)
print(submission.head())
