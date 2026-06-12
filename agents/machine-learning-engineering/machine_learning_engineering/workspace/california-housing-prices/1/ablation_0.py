import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Paths
INPUT_DIR = "./input"
train_path = os.path.join(INPUT_DIR, "train.csv")

# Load data
train_df = pd.read_csv(train_path)
target_col = "median_house_value"

# Same holdout split as the base solution
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def eval_variant(variant_name, build_graph_fn):
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    pred_graph = build_graph_fn(X_train, y_train)
    learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {rmse}")
    return rmse

base_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=2000,
    depth=8,
    learning_rate=0.03,
    random_seed=42,
    verbose=0,
)

def baseline_graph(X_train, y_train):
    vectorizer = skrub.TableVectorizer()
    return X_train.skb.apply(vectorizer).skb.apply(base_model, y=y_train)

def no_rooms_bedrooms_graph(X_train, y_train):
    # Redundancy ablation: drop one strongly correlated size/count column pair
    X_red = X_train.skb.apply(skrub.DropCols(cols=["total_bedrooms"]))
    vectorizer = skrub.TableVectorizer()
    return X_red.skb.apply(vectorizer).skb.apply(base_model, y=y_train)

def room_ratio_graph(X_train, y_train):
    # Structural feature ablation: add a ratio feature suggested by correlated count columns
    @skrub.deferred
    def add_room_ratios(df):
        out = df.copy()
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (
            (out["total_rooms"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
        return out

    X_fe = X_train.skb.apply_func(add_room_ratios)
    vectorizer = skrub.TableVectorizer()
    return X_fe.skb.apply(vectorizer).skb.apply(base_model, y=y_train)

scores = {}
scores["baseline"] = eval_variant("baseline", baseline_graph)
scores["drop_total_bedrooms"] = eval_variant("drop_total_bedrooms", no_rooms_bedrooms_graph)
scores["rooms_per_household"] = eval_variant("rooms_per_household", room_ratio_graph)

best_variant = min(scores, key=scores.get)
worst_variant = max(scores, key=scores.get)

print(f"Best ablation variant: {best_variant} | RMSE: {scores[best_variant]}")
print(f"Most harmful ablation: {worst_variant} | RMSE: {scores[worst_variant]}")