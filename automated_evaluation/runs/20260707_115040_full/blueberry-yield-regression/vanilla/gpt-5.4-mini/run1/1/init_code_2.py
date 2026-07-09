
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

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

model = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    loss_function="MAE",
    random_seed=SEED,
    verbose=200
)

model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)

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
