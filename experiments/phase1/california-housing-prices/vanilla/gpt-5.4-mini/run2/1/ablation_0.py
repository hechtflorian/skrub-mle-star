import os
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

# Load training data only
train_path = os.path.join("./input", "train.csv")
train = pd.read_csv(train_path)

X = train.drop(columns=["median_house_value"])
y = train["median_house_value"]

# Hold-out validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def train_and_eval(model_name, model):
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
    pred = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, pred)
    print(f"{model_name}: RMSE = {rmse:.6f}")
    return rmse

# Baseline: original configuration
baseline_model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42
)

# Ablation 1: remove early best model selection
no_best_model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42
)

# Ablation 2: simpler model depth
shallower_model = CatBoostRegressor(
    depth=4,
    learning_rate=0.05,
    iterations=3000,
    loss_function="RMSE",
    eval_metric="RMSE",
    random_seed=42
)

# Evaluate baseline
baseline_rmse = train_and_eval("Baseline (depth=8, use_best_model=True)", baseline_model)

# Evaluate ablation without best model selection
# Note: CatBoost use_best_model is controlled in fit; here we disable it explicitly.
no_best_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=False, verbose=False)
no_best_pred = no_best_model.predict(X_val)
no_best_rmse = root_mean_squared_error(y_val, no_best_pred)
print(f"Ablation 1 (use_best_model=False): RMSE = {no_best_rmse:.6f}")

# Evaluate ablation with smaller depth
shallower_rmse = train_and_eval("Ablation 2 (depth=4)", shallower_model)

# Compare contributions
results = {
    "baseline": baseline_rmse,
    "no_best_model": no_best_rmse,
    "shallower_depth": shallower_rmse,
}

print("\nAblation impact (higher positive delta means worse performance):")
print(f"use_best_model disabled: {no_best_rmse - baseline_rmse:+.6f}")
print(f"depth reduced to 4:      {shallower_rmse - baseline_rmse:+.6f}")

worst_change = max(
    [("use_best_model disabled", no_best_rmse - baseline_rmse),
     ("depth reduced to 4", shallower_rmse - baseline_rmse)],
    key=lambda x: x[1]
)

print(f"\nPart contributing most to performance: {worst_change[0]}")