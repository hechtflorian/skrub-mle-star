
import os
import random
import sys
import subprocess
import numpy as np
import pandas as pd

# Ensure lightgbm is available
try:
    from lightgbm import LGBMRegressor
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target_col = "median_house_value"
X = train.drop(columns=[target_col]).copy()
y = train[target_col].copy()

# Fill missing values using training medians
median_values = X.median(numeric_only=True)
X = X.fillna(median_values)
test = test.fillna(median_values)

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    objective="regression",
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
)

val_preds = model.predict(X_val)

# Compatibility with older sklearn versions that do not support squared=False
mse = mean_squared_error(y_val, val_preds)
val_rmse = np.sqrt(mse)

test_preds = model.predict(test)

submission = pd.DataFrame({target_col: test_preds})
submission.to_csv("submission.csv", index=False)

print(f"Final Validation Performance: {val_rmse}")
