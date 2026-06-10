import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

TARGET = "median_house_value"
DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")

train_df = pd.read_csv(TRAIN_PATH)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def fill_missing_numeric(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
    return df

def add_ratio_features(df):
    df = df.copy()
    eps = 1e-9
    df["rooms_per_household"] = df["total_rooms"] / (df["households"] + eps)
    df["bedrooms_per_room"] = df["total_bedrooms"] / (df["total_rooms"] + eps)
    df["population_per_household"] = df["population"] / (df["households"] + eps)
    return df

def build_and_score(df_train, df_valid, use_missing_fill=True, use_ratio_features=True, drop_redundant=False):
    data = skrub.var("data", df_train)
    X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    if use_missing_fill:
        X = X.skb.apply_func(fill_missing_numeric)
    if use_ratio_features:
        X = X.skb.apply_func(add_ratio_features)
    if drop_redundant:
        X = X.skb.apply(skrub.DropCols(cols=["households"]))

    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

    pred = X.skb.apply(model, y=y)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": df_valid})
    rmse = mean_squared_error(df_valid[TARGET], valid_pred) ** 0.5
    return rmse

results = {}

results["baseline"] = build_and_score(
    train_part, valid_part, use_missing_fill=True, use_ratio_features=True, drop_redundant=False
)
results["no_ratio_features"] = build_and_score(
    train_part, valid_part, use_missing_fill=True, use_ratio_features=False, drop_redundant=False
)
results["no_missing_fill"] = build_and_score(
    train_part, valid_part, use_missing_fill=False, use_ratio_features=True, drop_redundant=False
)
results["drop_households"] = build_and_score(
    train_part, valid_part, use_missing_fill=True, use_ratio_features=True, drop_redundant=True
)

baseline = results["baseline"]
print(f"Ablation[baseline] RMSE: {baseline:.6f}")
for name, score in results.items():
    if name == "baseline":
        continue
    delta = score - baseline
    print(f"Ablation[{name}] RMSE: {score:.6f} | delta_vs_baseline: {delta:+.6f}")

best_variant = min(results, key=results.get)
best_score = results[best_variant]
worst_variant = max(results, key=results.get)
worst_score = results[worst_variant]

print(f"Final Validation Performance: {baseline}")
print(f"Best ablation variant: {best_variant} | RMSE: {best_score:.6f}")
print(f"Worst ablation variant: {worst_variant} | RMSE: {worst_score:.6f}")

importance = {
    name: results[name] - baseline for name in results if name != "baseline"
}
most_important = min(importance, key=lambda k: importance[k])
least_important = max(importance, key=lambda k: importance[k])

print(
    f"Most helpful part of the code: {most_important} "
    f"(RMSE change vs baseline: {importance[most_important]:+.6f})"
)
print(
    f"Least helpful / harmful part of the code: {least_important} "
    f"(RMSE change vs baseline: {importance[least_important]:+.6f})"
)