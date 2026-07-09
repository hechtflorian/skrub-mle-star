
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

df = pd.read_csv(train_path)
test = pd.read_csv(test_path)

X = df.drop(columns=["id", "yield"])
y = df["yield"]
X_test = test.drop(columns=["id"])

# Same holdout split used for both base learners
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)

# =========================
# Model A: Original single holdout CatBoost seed-ensemble
# =========================
seeds_a = [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]
test_preds_a = []
va_preds_a = []

for seed in seeds_a:
    model_a = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        random_seed=seed,
        verbose=200
    )

    model_a.fit(
        X_tr,
        y_tr,
        eval_set=(X_va, y_va),
        use_best_model=True,
        early_stopping_rounds=200
    )

    va_pred_a = model_a.predict(X_va)
    va_preds_a.append(va_pred_a)

    test_pred_a = model_a.predict(X_test)
    test_preds_a.append(test_pred_a)

va_pred_a = np.mean(np.column_stack(va_preds_a), axis=1)
mae_a = mean_absolute_error(y_va, va_pred_a)
print(f"Model A Validation Performance: {mae_a}")

test_pred_a = np.mean(np.column_stack(test_preds_a), axis=1)

# =========================
# Model B: Fold-based CatBoost OOF/seeded version
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_pred_b = np.zeros(len(X))
test_preds_b = []

# Use a compact seed ensemble inside each fold to keep logic stable
seeds_b = [SEED, SEED + 1, SEED + 2]

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    X_fold_tr, X_fold_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_fold_tr, y_fold_va = y.iloc[tr_idx], y.iloc[va_idx]

    fold_test_preds = []

    for seed in seeds_b:
        model_b = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.03,
            depth=6,
            loss_function="MAE",
            random_seed=seed,
            verbose=200
        )

        model_b.fit(
            X_fold_tr,
            y_fold_tr,
            eval_set=(X_fold_va, y_fold_va),
            use_best_model=True,
            early_stopping_rounds=200
        )

        oof_pred_b[va_idx] += model_b.predict(X_fold_va) / len(seeds_b)
        fold_test_preds.append(model_b.predict(X_test))

    test_preds_b.append(np.mean(np.column_stack(fold_test_preds), axis=1))

mae_b = mean_absolute_error(y, oof_pred_b)
print(f"Model B Validation Performance: {mae_b}")

test_pred_b = np.mean(np.column_stack(test_preds_b), axis=1)

# =========================
# Ensemble at prediction level
# =========================
# Weighted average based on validation MAE; lower MAE gets slightly higher weight
if mae_a < mae_b:
    weight_a, weight_b = 0.6, 0.4
elif mae_b < mae_a:
    weight_a, weight_b = 0.4, 0.6
else:
    weight_a, weight_b = 0.5, 0.5

blended_va = weight_a * va_pred_a + weight_b * oof_pred_b[:len(y_va)]

# Tiny stacking layer using validation predictions from both models
# Train on the holdout subset where both model predictions are available
stack_X_va = np.column_stack([va_pred_a, oof_pred_b[:len(y_va)]])
stack_y_va = y_va.values

stack_model = LinearRegression()
stack_model.fit(stack_X_va, stack_y_va)

stack_X_test = np.column_stack([test_pred_a, test_pred_b])
stack_test_pred = stack_model.predict(stack_X_test)

# Final blend: use simple weighted average as the main ensemble, with stacking as a light calibrator
final_test_pred = 0.5 * (weight_a * test_pred_a + weight_b * test_pred_b) + 0.5 * stack_test_pred

# Optional clipping only if safe; keep disabled unless obviously needed
# final_test_pred = np.clip(final_test_pred, 0, None)

final_validation_pred = 0.5 * blended_va + 0.5 * stack_model.predict(stack_X_va)
final_validation_score = mean_absolute_error(y_va, final_validation_pred)
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({
    "id": test["id"],
    "yield": final_test_pred
})
submission.to_csv("submission.csv", index=False)
