
import os
import random
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

TARGET = "Rings"
ID_COL = "id"

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Basic feature engineering
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missing-safe numeric conversion
    for col in df.columns:
        if col not in [ID_COL, "Sex"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Geometry/weight interactions
    df["Length_Diameter"] = df["Length"] * df["Diameter"]
    df["Length_Height"] = df["Length"] * df["Height"]
    df["Diameter_Height"] = df["Diameter"] * df["Height"]
    df["Volume_approx"] = df["Length"] * df["Diameter"] * df["Height"]

    df["Whole_weight_ratio_1"] = df["Whole weight"] / (df["Shell weight"] + 1e-6)
    df["Whole_weight_ratio_2"] = df["Whole weight"] / (df["Whole weight.1"] + 1e-6)
    df["Whole_weight_ratio_3"] = df["Whole weight"] / (df["Whole weight.2"] + 1e-6)
    df["Shell_to_length"] = df["Shell weight"] / (df["Length"] + 1e-6)
    df["Shell_to_diameter"] = df["Shell weight"] / (df["Diameter"] + 1e-6)

    df["Height_to_length"] = df["Height"] / (df["Length"] + 1e-6)
    df["Height_to_diameter"] = df["Height"] / (df["Diameter"] + 1e-6)

    df["Sum_dims"] = df["Length"] + df["Diameter"] + df["Height"]
    df["Sum_weights"] = df["Whole weight"] + df["Whole weight.1"] + df["Whole weight.2"] + df["Shell weight"]

    return df

train_fe = add_features(train)
test_fe = add_features(test)

cat_cols = ["Sex"]

# One-hot encoding
full = pd.concat([train_fe.drop(columns=[TARGET]), test_fe], axis=0, ignore_index=True)
full = pd.get_dummies(full, columns=cat_cols, dummy_na=True)

X_train = full.iloc[: len(train_fe)].copy()
X_test = full.iloc[len(train_fe) :].copy()
y = train_fe[TARGET].astype(float).values

# Ensure no ID leakage
if ID_COL in X_train.columns:
    X_train = X_train.drop(columns=[ID_COL])
if ID_COL in X_test.columns:
    X_test = X_test.drop(columns=[ID_COL])

# Try lightgbm first; fall back to sklearn if unavailable
use_lgb = True
try:
    import lightgbm as lgb
except ModuleNotFoundError:
    use_lgb = False

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import Ridge

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

oof_pred = np.zeros(len(X_train), dtype=float)
test_pred = np.zeros(len(X_test), dtype=float)

if use_lgb:
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,          # keep subsampling
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=SEED + fold,
            n_jobs=-1,
        )

        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        va_pred = model.predict(X_va, num_iteration=model.best_iteration_)
        te_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        oof_pred[va_idx] = va_pred
        test_pred += te_pred / kf.n_splits
else:
    # Fast fallback model to avoid long runtime
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train), 1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = Ridge(alpha=2.0, random_state=SEED)
        model.fit(X_tr, y_tr)

        va_pred = model.predict(X_va)
        te_pred = model.predict(X_test)

        oof_pred[va_idx] = va_pred
        test_pred += te_pred / kf.n_splits

# Clip to valid range
oof_pred = np.clip(oof_pred, 0, None)
test_pred = np.clip(test_pred, 0, None)

# RMSLE-compatible rounding only for final evaluation/reporting
oof_pred_round = np.clip(np.rint(oof_pred), 0, None)
final_validation_score = np.sqrt(mean_squared_log_error(y, oof_pred_round))
print(f"Final Validation Performance: {final_validation_score}")

test_pred_round = np.clip(np.rint(test_pred), 0, None).astype(int)

submission = pd.DataFrame({
    ID_COL: test[ID_COL],
    TARGET: test_pred_round
})
submission.to_csv("submission.csv", index=False)
