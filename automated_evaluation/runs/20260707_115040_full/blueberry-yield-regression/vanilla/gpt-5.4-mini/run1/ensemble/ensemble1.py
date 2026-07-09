
import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
from scipy.stats import rankdata

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

df = pd.read_csv(train_path)

X = df.drop(columns=["id", "yield"])
y = df["yield"].values

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED)

test = pd.read_csv(test_path)
X_test = test.drop(columns=["id"])

# Model A: keep original holdout setup with seed ensemble
seeds_a = [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]
a_va_preds = []
a_test_preds = []

for seed in seeds_a:
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

    a_va_preds.append(model.predict(X_va))
    a_test_preds.append(model.predict(X_test))

a_va_pred = np.mean(np.column_stack(a_va_preds), axis=1)
a_test_pred = np.mean(np.column_stack(a_test_preds), axis=1)

# Model B: OOF predictions using KFold + seed ensemble
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
seeds_b = [SEED, SEED + 1, SEED + 2, SEED + 3, SEED + 4]

b_va_pred = np.zeros(len(X_va))
b_test_preds_folds = []

X_va_np = X_va.reset_index(drop=True)
X_tr_np = X_tr.reset_index(drop=True)
X_test_np = X_test.copy()

# For B, we create OOF on the holdout validation set by training only on X_tr and predicting X_va
# using internal folds to generate more diverse predictions, then averaging across folds.
# This keeps the model structure simple while providing an OOF-like validation prediction.
for seed in seeds_b:
    fold_va_pred = np.zeros(len(X_va))
    fold_test_pred = np.zeros(len(X_test))

    for tr_idx, val_idx in kf.split(X_tr_np):
        X_fold_tr = X_tr_np.iloc[tr_idx]
        y_fold_tr = y_tr[tr_idx]
        X_fold_va = X_tr_np.iloc[val_idx]
        y_fold_va = y_tr[val_idx]

        model = CatBoostRegressor(
            iterations=5000,
            learning_rate=0.03,
            depth=6,
            loss_function="MAE",
            random_seed=seed,
            verbose=200
        )

        model.fit(
            X_fold_tr,
            y_fold_tr,
            eval_set=(X_fold_va, y_fold_va),
            use_best_model=True,
            early_stopping_rounds=200
        )

        # For validation predictions, aggregate on the external holdout set
        fold_va_pred += model.predict(X_va_np) / kf.n_splits
        fold_test_pred += model.predict(X_test_np) / kf.n_splits

    b_va_pred += fold_va_pred / len(seeds_b)
    b_test_preds_folds.append(fold_test_pred / len(seeds_b))

b_test_pred = np.mean(np.column_stack(b_test_preds_folds), axis=1)

# Rank-based blending
def to_rank(x):
    return rankdata(x, method="average") / len(x)

a_va_rank = to_rank(a_va_pred)
b_va_rank = to_rank(b_va_pred)
a_test_rank = to_rank(a_test_pred)
b_test_rank = to_rank(b_test_pred)

# Equal weights as requested
blend_va_rank = 0.5 * a_va_rank + 0.5 * b_va_rank
blend_test_rank = 0.5 * a_test_rank + 0.5 * b_test_rank

# Inverse-rank mapping using empirical target distribution
sorted_y = np.sort(y)

def inverse_rank_map(ranks, sorted_targets):
    n = len(sorted_targets)
    idx = np.clip((ranks * (n - 1)).astype(int), 0, n - 1)
    return sorted_targets[idx]

final_va_pred = inverse_rank_map(blend_va_rank, sorted_y)
final_test_pred = inverse_rank_map(blend_test_rank, sorted_y)

final_va_mae = mean_absolute_error(y_va, final_va_pred)
print(f"Final Validation Performance: {final_va_mae}")

submission = pd.DataFrame({
    "id": test["id"],
    "yield": final_test_pred
})
submission.to_csv("submission.csv", index=False)
