
import os
import sys
import subprocess
import random
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

random.seed(42)
np.random.seed(42)

train_path = os.path.join(".", "input", "train.csv")
train = pd.read_csv(train_path)

def add_date_features(df, use_date_features=True):
    df = df.copy()
    if use_date_features:
        df["Open Date"] = pd.to_datetime(df["Open Date"], format="%m/%d/%Y")
        df["open_year"] = df["Open Date"].dt.year
        df["open_month"] = df["Open Date"].dt.month
        df["open_day"] = df["Open Date"].dt.day
        df["open_weekday"] = df["Open Date"].dt.weekday
        reference_date = pd.Timestamp("2015-01-01")
        df["restaurant_age_days"] = (reference_date - df["Open Date"]).dt.days
        df.drop(columns=["Open Date"], inplace=True)
    else:
        df.drop(columns=["Open Date"], inplace=True)
    return df

def evaluate_ablation(name, use_date_features=True, use_categoricals=True):
    df = add_date_features(train, use_date_features=use_date_features)

    X = df.drop(columns=["revenue"])
    y = df["revenue"].copy()

    cat_cols = ["City", "City Group", "Type"]
    if not use_categoricals:
        X = X.drop(columns=cat_cols)
        cat_idx = []
    else:
        cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=False
    )

    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "eval_set": (X_valid, y_valid),
        "use_best_model": True
    }
    if use_categoricals and len(cat_idx) > 0:
        fit_kwargs["cat_features"] = cat_idx

    model.fit(**fit_kwargs)

    valid_pred = model.predict(X_valid)
    rmse = float(np.sqrt(mean_squared_error(y_valid, valid_pred)))
    print(f"{name}: RMSE = {rmse:.6f}")
    return rmse

results = {}
results["baseline"] = evaluate_ablation(
    "baseline",
    use_date_features=True,
    use_categoricals=True
)
results["no_date_features"] = evaluate_ablation(
    "no_date_features",
    use_date_features=False,
    use_categoricals=True
)
results["no_categorical_features"] = evaluate_ablation(
    "no_categorical_features",
    use_date_features=True,
    use_categoricals=False
)

baseline_rmse = results["baseline"]
ablation_deltas = {
    k: v - baseline_rmse
    for k, v in results.items()
    if k != "baseline"
}

worst_ablation = max(ablation_deltas, key=ablation_deltas.get)
print(f"Most important part: {worst_ablation} changed RMSE by {ablation_deltas[worst_ablation]:.6f} compared to baseline.")
