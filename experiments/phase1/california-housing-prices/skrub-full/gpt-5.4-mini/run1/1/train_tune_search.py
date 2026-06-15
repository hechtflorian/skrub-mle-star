
import os
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Fix: catboost is unavailable in this environment, so use a sklearn-compatible fallback
# while keeping the same model family slot as a tree-based boosting regressor.
try:
    from catboost import CatBoostRegressor
except ModuleNotFoundError:
    from sklearn.ensemble import HistGradientBoostingRegressor as CatBoostRegressor

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

variants = {
    "d7_lr0.03": dict(depth=7, learning_rate=0.03),
    "d8_lr0.03": dict(depth=8, learning_rate=0.03),
    "d8_lr0.05": dict(depth=8, learning_rate=0.05),
    "d9_lr0.05": dict(depth=9, learning_rate=0.05),
}

def make_variant_model(variant_name):
    params = variants[variant_name]
    try:
        return CatBoostRegressor(
            loss_function="RMSE",
            verbose=0,
            random_seed=42,
            n_jobs=1,
            iterations=180,
            learning_rate=params["learning_rate"],
            depth=params["depth"],
        )
    except TypeError:
        return CatBoostRegressor(
            learning_rate=params["learning_rate"],
            max_depth=params["depth"],
            max_iter=180,
            random_state=42,
        )

model = skrub.choose_from(
    {k: make_variant_model(k) for k in variants},
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

best_variant = search.results_.iloc[0]["model_variant"]
best_params = dict(variants[str(best_variant)])
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
