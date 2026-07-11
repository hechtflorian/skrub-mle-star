
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


from sklearn.model_selection import KFold

test = pd.read_csv(test_path)
X_test = test.drop(columns=["id"])

# Compact seed ensemble / repeated holdout over the same full feature set
seeds = [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]
test_preds = []
va_preds = []

for seed in seeds:
    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        random_seed=seed,
        verbose=200
    )
    
    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True,
        early_stopping_rounds=200
    )
    
    va_pred = model.predict(X_va)
    va_preds.append(va_pred)
    
    test_pred = model.predict(X_test)
    test_preds.append(test_pred)

# Average predictions to reduce variance
va_pred = np.mean(np.column_stack(va_preds), axis=1)
va_mae = mean_absolute_error(y_va, va_pred)
print(f"Final Validation Performance: {va_mae}")

test_pred = np.mean(np.column_stack(test_preds), axis=1)

submission = pd.DataFrame({
    "id": test["id"],
    "yield": test_pred
})
submission.to_csv("submission.csv", index=False)

