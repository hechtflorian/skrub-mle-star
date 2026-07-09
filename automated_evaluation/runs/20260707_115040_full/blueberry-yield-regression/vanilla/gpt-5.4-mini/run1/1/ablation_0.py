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
df = pd.read_csv(train_path)

X = df.drop(columns=["id", "yield"])
y = df["yield"]

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)

def train_and_eval(name, model_kwargs, use_best_model=True, early_stopping_rounds=200):
    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        random_seed=SEED,
        verbose=0,
        **model_kwargs
    )
    fit_kwargs = dict(
        X=X_tr,
        y=y_tr,
        eval_set=(X_va, y_va),
        use_best_model=use_best_model,
        verbose=0
    )
    if early_stopping_rounds is not None:
        fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
    model.fit(**fit_kwargs)
    pred = model.predict(X_va)
    mae = mean_absolute_error(y_va, pred)
    print(f"{name}: MAE = {mae:.6f}")
    return mae

base_mae = train_and_eval("Baseline", model_kwargs={})

# Ablation 1: disable early stopping / best model usage
no_best_mae = train_and_eval(
    "Ablation 1 - No best model / no early stopping",
    model_kwargs={},
    use_best_model=False,
    early_stopping_rounds=None
)

# Ablation 2: remove a strong feature block (fruit-related features)
drop_cols = [c for c in ["fruitset", "fruitmass", "seeds"] if c in X.columns]
X_tr_drop = X_tr.drop(columns=drop_cols)
X_va_drop = X_va.drop(columns=drop_cols)

model_drop = CatBoostRegressor(
    iterations=5000,
    learning_rate=0.03,
    depth=6,
    loss_function="MAE",
    random_seed=SEED,
    verbose=0
)
model_drop.fit(
    X_tr_drop, y_tr,
    eval_set=(X_va_drop, y_va),
    use_best_model=True,
    early_stopping_rounds=200,
    verbose=0
)
drop_pred = model_drop.predict(X_va_drop)
drop_mae = mean_absolute_error(y_va, drop_pred)
print(f"Ablation 2 - Drop fruit features {drop_cols}: MAE = {drop_mae:.6f}")

# Compare deltas
results = {
    "Baseline": base_mae,
    "No best model / no early stopping": no_best_mae,
    f"Drop fruit features {drop_cols}": drop_mae,
}

best_name = min(results, key=results.get)
print("\nMAE comparison (lower is better):")
for k, v in results.items():
    delta = v - base_mae
    print(f"{k}: {v:.6f} (delta vs baseline: {delta:+.6f})")

print(f"\nMost important part for performance: {best_name} (lowest MAE = {results[best_name]:.6f})")