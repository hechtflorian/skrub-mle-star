
import sys
import subprocess
import json

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

train_df = pd.read_csv("./input/train.csv")
target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


@skrub.deferred
def add_ratio_features(df):
    out = df.copy()
    for numer_col, denom_col, out_col in [
        ("total_rooms", "households", "rooms_per_household"),
        ("total_bedrooms", "households", "bedrooms_per_household"),
        ("population", "households", "population_per_household"),
    ]:
        if numer_col in out.columns and denom_col in out.columns:
            denom = out[denom_col].replace(0, np.nan)
            out[out_col] = (
                out[numer_col] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_ratio_features)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

variants = {
    "depth6_lr0.05_it300": dict(depth=6, learning_rate=0.05, iterations=300),
    "depth7_lr0.03_it300": dict(depth=7, learning_rate=0.03, iterations=300),
    "depth7_lr0.05_it200": dict(depth=7, learning_rate=0.05, iterations=200),
    "depth8_lr0.03_it200": dict(depth=8, learning_rate=0.03, iterations=200),
}

model = skrub.choose_from(
    {
        k: CatBoostRegressor(
            **p,
            verbose=0,
            random_seed=42,
            thread_count=1,
        )
        for k, p in variants.items()
    },
    name="model_variant",
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

search = pred.skb.make_randomized_search(
    n_iter=4, n_jobs=1, random_state=42, fitted=True
)
search.fit({"data": train_part})

best_learner = search.best_learner_
valid_pred = best_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

best_variant = search.best_params_.get("model_variant", None)
if isinstance(best_variant, np.generic):
    best_variant = best_variant.item()

best_params = {}
if best_variant in variants:
    best_params = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in variants[best_variant].items()}
else:
    best_params = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in variants["depth6_lr0.05_it300"].items()}

print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
