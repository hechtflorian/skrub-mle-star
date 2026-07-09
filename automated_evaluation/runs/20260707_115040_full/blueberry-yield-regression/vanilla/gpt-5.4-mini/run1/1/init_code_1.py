
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

df = pd.read_csv(train_path)

X = df.drop(columns=["id", "yield"])
y = df["yield"]

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)

model = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_va, y_va)],
    eval_metric="mae",
)

va_pred = model.predict(X_va)
va_mae = mean_absolute_error(y_va, va_pred)
print(f"Final Validation Performance: {va_mae}")

test = pd.read_csv(test_path)
test_pred = model.predict(test.drop(columns=["id"]))

submission = pd.DataFrame({
    "id": test["id"],
    "yield": test_pred
})
submission.to_csv("submission.csv", index=False)
