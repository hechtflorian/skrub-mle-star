
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ablation_0.py.py

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def build_and_score(df_train, df_valid, variant_name, use_ratio_features=False):
    data_train = skrub.var("data", df_train)

    if use_ratio_features:
        @skrub.deferred
        def add_ratio_features(df):
            out = df.copy()
            if "households" in out.columns:
                denom = out["households"].replace(0, np.nan)
                out["rooms_per_household"] = (
                    out["total_rooms"] / denom
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                out["bedrooms_per_household"] = (
                    out["total_bedrooms"] / denom
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if "population" in out.columns and "households" in out.columns:
                denom = out["households"].replace(0, np.nan)
                out["population_per_household"] = (
                    out["population"] / denom
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return out

        data_train = data_train.skb.apply_func(add_ratio_features)

    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": df_valid})
    score = mean_squared_error(df_valid[target_col], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {score}")
    return score

baseline_score = build_and_score(train_part, valid_part, "baseline", use_ratio_features=False)
ablated_score = build_and_score(train_part, valid_part, "ratio_features", use_ratio_features=True)

best_variant = "baseline" if baseline_score <= ablated_score else "ratio_features"
best_score = min(baseline_score, ablated_score)

print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")
print(f"Final Validation Performance: {best_score}")
