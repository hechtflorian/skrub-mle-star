import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

train_path = "./input/train.csv"
train = pd.read_csv(train_path)

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# Preprocessing
X = pd.get_dummies(train.drop(columns=["Rings"]))
y = np.log1p(train["Rings"].astype(float))

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)

valid_true = np.expm1(y_valid).clip(0, None)

# ----------------------------
# Baseline: LGBM + CatBoost ensemble
# ----------------------------
lgb_model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

lgb_pred = np.expm1(lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration_)).clip(0, None)

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    verbose=False,
    random_seed=SEED
)

cat_model.fit(
    X_train,
    y_train,
    eval_set=(X_valid, y_valid),
    use_best_model=True
)

cat_pred = np.expm1(cat_model.predict(X_valid)).clip(0, None)

baseline_pred = 0.5 * lgb_pred + 0.5 * cat_pred
baseline_score = rmsle(valid_true, baseline_pred)
print(f"Baseline (LGBM + CatBoost ensemble) RMSLE: {baseline_score:.6f}")

# ----------------------------
# Ablation 1: Disable CatBoost, keep only LightGBM
# ----------------------------
lgb_only_pred = lgb_pred
lgb_only_score = rmsle(valid_true, lgb_only_pred)
print(f"Ablation 1 (LightGBM only) RMSLE: {lgb_only_score:.6f} | Delta vs baseline: {lgb_only_score - baseline_score:+.6f}")

# ----------------------------
# Ablation 2: Disable log1p target transform, keep LightGBM only
# ----------------------------
y_raw = train["Rings"].astype(float)
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(
    X, y_raw, test_size=0.2, random_state=SEED, shuffle=True
)

lgb_model_raw = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)

lgb_model_raw.fit(
    X_train2,
    y_train2,
    eval_set=[(X_valid2, y_valid2)],
    eval_metric="rmse",
    callbacks=[lgb.early_stopping(200, verbose=False)]
)

raw_pred = lgb_model_raw.predict(X_valid2)
raw_score = rmsle(y_valid2, raw_pred)
print(f"Ablation 2 (LightGBM without log1p target transform) RMSLE: {raw_score:.6f} | Delta vs baseline: {raw_score - baseline_score:+.6f}")

# ----------------------------
# Ablation 3: Disable early stopping, train LightGBM for fixed iterations
# ----------------------------
lgb_model_no_es = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED
)

lgb_model_no_es.fit(X_train, y_train)
no_es_pred = np.expm1(lgb_model_no_es.predict(X_valid)).clip(0, None)
no_es_score = rmsle(valid_true, no_es_pred)
print(f"Ablation 3 (LightGBM without early stopping) RMSLE: {no_es_score:.6f} | Delta vs baseline: {no_es_score - baseline_score:+.6f}")

# ----------------------------
# Which part contributes the most?
# The larger the positive delta, the more that component helps.
# ----------------------------
ablation_results = {
    "LightGBM only": lgb_only_score,
    "No log1p target transform": raw_score,
    "No early stopping": no_es_score,
}

best_contributor = min(ablation_results, key=ablation_results.get)
best_improvement = baseline_score - ablation_results[best_contributor]

print("\nAblation summary:")
for name, score in ablation_results.items():
    print(f"- {name}: RMSLE = {score:.6f}, Delta = {score - baseline_score:+.6f}")

print(
    f"\nMost important component for performance: {best_contributor} "
    f"(removing it worsens RMSLE by {abs(best_improvement):.6f})"
)