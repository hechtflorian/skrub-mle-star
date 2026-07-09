
import os
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from scipy.stats import rankdata

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Preprocessing
X = pd.get_dummies(train.drop(columns=["Rings"]))
y = np.log1p(train["Rings"].astype(float))
X_test = pd.get_dummies(test)

X, X_test = X.align(X_test, join="left", axis=1, fill_value=0)

n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)

# Solution 1 OOF/Test
oof_lgb_log = np.zeros(len(X))
oof_cat_log = np.zeros(len(X))
test_lgb_log = np.zeros(len(X_test))
test_cat_log = np.zeros(len(X_test))
lgb_best_iterations = []

# Solution 2 OOF/Test (feature-engineered tree model source)
oof_sol2 = np.zeros(len(X))
test_sol2 = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # -------------------------
    # Solution 1: LightGBM
    # -------------------------
    lgb_model = lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.02,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED + fold
    )

    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )

    best_iter = lgb_model.best_iteration_ if lgb_model.best_iteration_ is not None else lgb_model.n_estimators
    lgb_best_iterations.append(best_iter)

    oof_lgb_log[valid_idx] = lgb_model.predict(X_valid, num_iteration=best_iter)
    test_lgb_log += lgb_model.predict(X_test, num_iteration=best_iter) / n_splits

    # -------------------------
    # Solution 1: CatBoost
    # -------------------------
    cat_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        verbose=False,
        random_seed=SEED + fold
    )

    cat_model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=True
    )

    oof_cat_log[valid_idx] = cat_model.predict(X_valid)
    test_cat_log += cat_model.predict(X_test) / n_splits

    # -------------------------
    # Solution 2: complementary feature-engineered tree model
    # Keep it minimal and compatible with the ensemble plan.
    # -------------------------
    X_train_2 = X_train.copy()
    X_valid_2 = X_valid.copy()
    X_test_2 = X_test.copy()

    # Low-risk handcrafted interactions using age-related signal
    if "Age" in X_train_2.columns and "ShellWeight" in X_train_2.columns:
        X_train_2["Age_ShellWeight"] = X_train_2["Age"] * X_train_2["ShellWeight"]
        X_valid_2["Age_ShellWeight"] = X_valid_2["Age"] * X_valid_2["ShellWeight"]
        X_test_2["Age_ShellWeight"] = X_test_2["Age"] * X_test_2["ShellWeight"]

    if "Length" in X_train_2.columns and "Diameter" in X_train_2.columns:
        X_train_2["Length_Diameter"] = X_train_2["Length"] * X_train_2["Diameter"]
        X_valid_2["Length_Diameter"] = X_valid_2["Length"] * X_valid_2["Diameter"]
        X_test_2["Length_Diameter"] = X_test_2["Length"] * X_test_2["Diameter"]

    if "WholeWeight" in X_train_2.columns and "ShuckedWeight" in X_train_2.columns:
        X_train_2["Whole_Shucked"] = X_train_2["WholeWeight"] - X_train_2["ShuckedWeight"]
        X_valid_2["Whole_Shucked"] = X_valid_2["WholeWeight"] - X_valid_2["ShuckedWeight"]
        X_test_2["Whole_Shucked"] = X_test_2["WholeWeight"] - X_test_2["ShuckedWeight"]

    # A slightly different tree setup for complementary behavior
    sol2_model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=SEED + 100 + fold
    )

    sol2_model.fit(
        X_train_2,
        y_train,
        eval_set=[(X_valid_2, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(150, verbose=False)]
    )

    best_iter_2 = sol2_model.best_iteration_ if sol2_model.best_iteration_ is not None else sol2_model.n_estimators
    oof_sol2[valid_idx] = sol2_model.predict(X_valid_2, num_iteration=best_iter_2)
    test_sol2 += sol2_model.predict(X_test_2, num_iteration=best_iter_2) / n_splits

# Convert Solution 1 to original scale
valid_true = np.expm1(y).clip(0, None)
oof_lgb = np.expm1(oof_lgb_log).clip(0, None)
oof_cat = np.expm1(oof_cat_log).clip(0, None)

# Blend Solution 1 components on original scale
oof_sol1 = 0.8 * oof_lgb + 0.2 * oof_cat
test_sol1 = 0.8 * np.expm1(test_lgb_log).clip(0, None) + 0.2 * np.expm1(test_cat_log).clip(0, None)

# Solution 2 predictions are already on target scale; just clip negatives
oof_sol2 = np.clip(oof_sol2, 0, None)
test_sol2 = np.clip(test_sol2, 0, None)

# Optional rank averaging for robustness, then weighted average
oof_sol1_rank = rankdata(oof_sol1, method="average") / len(oof_sol1)
oof_sol2_rank = rankdata(oof_sol2, method="average") / len(oof_sol2)
test_sol1_rank = rankdata(test_sol1, method="average") / len(test_sol1)
test_sol2_rank = rankdata(test_sol2, method="average") / len(test_sol2)

# OOF-based weight search for the final blend
best_score = float("inf")
best_w = 0.7
for w in np.linspace(0.1, 0.9, 17):
    oof_blend_rank = w * oof_sol1_rank + (1.0 - w) * oof_sol2_rank
    oof_blend = w * oof_sol1 + (1.0 - w) * oof_sol2
    score = np.sqrt(mean_squared_log_error(valid_true, np.clip(oof_blend, 0, None)))
    if score < best_score:
        best_score = score
        best_w = w

# Final ensemble: blend original-scale predictions with the selected OOF weight
oof_pred = best_w * oof_sol1 + (1.0 - best_w) * oof_sol2
final_validation_score = np.sqrt(mean_squared_log_error(valid_true, np.clip(oof_pred, 0, None)))
print(f"Final Validation Performance: {final_validation_score}")

# Train final models on all data with averaged best iteration for LightGBM
avg_best_iter = int(np.mean(lgb_best_iterations)) if len(lgb_best_iterations) > 0 else 5000

lgb_final_model = lgb.LGBMRegressor(
    n_estimators=avg_best_iter,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)
lgb_final_model.fit(X, y)

cat_final_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    verbose=False,
    random_seed=SEED
)
cat_final_model.fit(X, y)

# Solution 2 final model
X_all_2 = X.copy()
X_test_2 = X_test.copy()
if "Age" in X_all_2.columns and "ShellWeight" in X_all_2.columns:
    X_all_2["Age_ShellWeight"] = X_all_2["Age"] * X_all_2["ShellWeight"]
    X_test_2["Age_ShellWeight"] = X_test_2["Age"] * X_test_2["ShellWeight"]
if "Length" in X_all_2.columns and "Diameter" in X_all_2.columns:
    X_all_2["Length_Diameter"] = X_all_2["Length"] * X_all_2["Diameter"]
    X_test_2["Length_Diameter"] = X_test_2["Length"] * X_test_2["Diameter"]
if "WholeWeight" in X_all_2.columns and "ShuckedWeight" in X_all_2.columns:
    X_all_2["Whole_Shucked"] = X_all_2["WholeWeight"] - X_all_2["ShuckedWeight"]
    X_test_2["Whole_Shucked"] = X_test_2["WholeWeight"] - X_test_2["ShuckedWeight"]

sol2_final_model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=SEED
)
sol2_final_model.fit(X_all_2, y)

# Test predictions and ensemble
lgb_test_pred_log = lgb_final_model.predict(X_test)
cat_test_pred_log = cat_final_model.predict(X_test)
sol2_test_pred = sol2_final_model.predict(X_test_2)

lgb_test_pred = np.expm1(lgb_test_pred_log).clip(0, None)
cat_test_pred = np.expm1(cat_test_pred_log).clip(0, None)
test_sol1 = 0.8 * lgb_test_pred + 0.2 * cat_test_pred
test_sol2 = np.clip(sol2_test_pred, 0, None)

# Final weighted average
test_pred = best_w * test_sol1 + (1.0 - best_w) * test_sol2
test_pred = np.clip(test_pred, 0, None)

submission = pd.DataFrame({"id": test["id"], "Rings": test_pred})
submission.to_csv("submission.csv", index=False)
