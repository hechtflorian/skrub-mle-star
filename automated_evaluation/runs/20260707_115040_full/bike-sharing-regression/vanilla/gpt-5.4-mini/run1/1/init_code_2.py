
import os
import sys
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import HistGradientBoostingRegressor

# Ensure xgboost is available if possible, but provide a fallback model
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ModuleNotFoundError:
    HAS_XGBOOST = False
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost", "-q"])
        from xgboost import XGBRegressor
        HAS_XGBOOST = True
    except Exception:
        HAS_XGBOOST = False

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["datetime"])
    df = df.copy()
    df["hour"] = dt.dt.hour
    df["day"] = dt.dt.day
    df["month"] = dt.dt.month
    df["year"] = dt.dt.year
    df["weekday"] = dt.dt.weekday
    df["dayofyear"] = dt.dt.dayofyear
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["weekend"] = (df["weekday"] >= 5).astype(int)
    return df

train = add_datetime_features(train)
test = add_datetime_features(test)

features = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "day", "month", "year", "weekday",
    "dayofyear", "weekofyear", "weekend"
]

train = train.sort_values("datetime").reset_index(drop=True)
split_idx = int(len(train) * 0.8)
train_part = train.iloc[:split_idx].copy()
valid_part = train.iloc[split_idx:].copy()

X_train = train_part[features]
y_train = train_part["count"].astype(float)
X_valid = valid_part[features]
y_valid = valid_part["count"].astype(float)

if HAS_XGBOOST:
    model = XGBRegressor(
        n_estimators=1500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
else:
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=500,
        l2_regularization=1.0,
        random_state=42,
        early_stopping=False
    )

model.fit(X_train, y_train)

valid_pred = model.predict(X_valid)
valid_pred = np.clip(valid_pred, 0, None)
final_validation_score = np.sqrt(mean_squared_log_error(y_valid, valid_pred))
print(f"Final Validation Performance: {final_validation_score}")

X_full = train[features]
y_full = train["count"].astype(float)
model.fit(X_full, y_full)

test_pred = model.predict(test[features])
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({
    "datetime": test["datetime"],
    "count": test_pred
})
submission.to_csv("submission.csv", index=False)

print(submission.head())
