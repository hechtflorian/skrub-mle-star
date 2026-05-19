
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# Load data
train = pd.read_csv("./input/train.csv")

# Features and target
X = train.drop(columns=["median_house_value"])
y = train["median_house_value"]

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline performance: XGBoost + LightGBM ensemble
def baseline():
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )

    xgb_val_preds = xgb_model.predict(X_val)
    lgb_val_preds = lgb_model.predict(X_val)
    ensemble_val_preds = (xgb_val_preds + lgb_val_preds) / 2
    rmse = mean_squared_error(y_val, ensemble_val_preds) ** 0.5
    return rmse

# Ablation 1: Disable early stopping for XGBoost
def ablation_early_stopping_xgb():
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )

    xgb_val_preds = xgb_model.predict(X_val)
    lgb_val_preds = lgb_model.predict(X_val)
    ensemble_val_preds = (xgb_val_preds + lgb_val_preds) / 2
    rmse = mean_squared_error(y_val, ensemble_val_preds) ** 0.5
    return rmse

# Ablation 2: Disable subsampling for XGBoost
def ablation_subsampling_xgb():
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        subsample=1.0,  # Disable subsampling
        colsample_bytree=1.0,  # Disable column subsampling
        random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )

    xgb_val_preds = xgb_model.predict(X_val)
    lgb_val_preds = lgb_model.predict(X_val)
    ensemble_val_preds = (xgb_val_preds + lgb_val_preds) / 2
    rmse = mean_squared_error(y_val, ensemble_val_preds) ** 0.5
    return rmse

# Ablation 3: Use only XGBoost (no ensemble)
def ablation_no_ensemble_lgb():
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        early_stopping_rounds=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)

    xgb_val_preds = xgb_model.predict(X_val)
    rmse = mean_squared_error(y_val, xgb_val_preds) ** 0.5
    return rmse

# Ablation 4: Use only LightGBM (no ensemble)
def ablation_no_ensemble_xgb():
    lgb_model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )

    lgb_val_preds = lgb_model.predict(X_val)
    rmse = mean_squared_error(y_val, lgb_val_preds) ** 0.5
    return rmse

# Run ablations
baseline_rmse = baseline()
ablation_1_rmse = ablation_early_stopping_xgb()
ablation_2_rmse = ablation_subsampling_xgb()
ablation_3_rmse = ablation_no_ensemble_lgb()
ablation_4_rmse = ablation_no_ensemble_xgb()

print(f"Baseline RMSE (XGBoost + LightGBM ensemble): {baseline_rmse}")
print(f"Ablation 1 RMSE (Disable early stopping for XGBoost): {ablation_1_rmse}")
print(f"Ablation 2 RMSE (Disable subsampling in XGBoost): {ablation_2_rmse}")
print(f"Ablation 3 RMSE (Only XGBoost, no ensemble): {ablation_3_rmse}")
print(f"Ablation 4 RMSE (Only LightGBM, no ensemble): {ablation_4_rmse}")

# Determine which part contributes most
performance_diff = {
    "Early stopping": baseline_rmse - ablation_1_rmse,
    "Subsampling": baseline_rmse - ablation_2_rmse,
    "Ensemble with LightGBM": baseline_rmse - ablation_3_rmse,
    "Ensemble with XGBoost": baseline_rmse - ablation_4_rmse
}

max_contribution = max(performance_diff, key=performance_diff.get)
print(f"\nThe part contributing most to performance is: {max_contribution} (Impact: {performance_diff[max_contribution]:.4f})")
