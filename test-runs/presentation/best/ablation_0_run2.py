import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

input_dir = "./input"
train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))

target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


@skrub.deferred
def add_ratio_features(df):
    out = df.copy()
    out["rooms_per_household"] = out["total_rooms"] / (out["households"] + 1e-9)
    out["bedrooms_per_room"] = out["total_bedrooms"] / (out["total_rooms"] + 1e-9)
    out["population_per_household"] = out["population"] / (out["households"] + 1e-9)
    return out


def evaluate_variant(variant_name, use_ratio_features=False, drop_redundant=False):
    data_train = skrub.var("data", train_part)

    if use_ratio_features:
        data_train = data_train.skb.apply_func(add_ratio_features)

    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    if drop_redundant:
        X_train = X_train.skb.apply(skrub.DropCols(cols=["households"]))

    model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=500,
        learning_rate=0.05,
        depth=8,
        random_seed=42,
        verbose=0,
    )

    pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)
    val_learner = pred.skb.make_learner(fitted=True)

    valid_env = {"data": valid_part}
    if use_ratio_features:
        valid_env = {"data": valid_part}
    valid_pred = val_learner.predict(valid_env)

    score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {score}")
    return score


scores = {}

scores["baseline"] = evaluate_variant(
    "baseline",
    use_ratio_features=False,
    drop_redundant=False,
)

scores["ratio_features"] = evaluate_variant(
    "ratio_features",
    use_ratio_features=True,
    drop_redundant=False,
)

scores["drop_households"] = evaluate_variant(
    "drop_households",
    use_ratio_features=True,
    drop_redundant=True,
)

best_variant = min(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")