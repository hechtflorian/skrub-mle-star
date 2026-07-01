
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"


@skrub.deferred
def add_housing_ratio_features(df):
    out = df.copy()
    if "households" in out.columns and "total_rooms" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "total_rooms" in out.columns and "total_bedrooms" in out.columns:
        denom = out["total_rooms"].replace(0, np.nan)
        out["bedrooms_per_room"] = (out["total_bedrooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "households" in out.columns and "population" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["population_per_household"] = (out["population"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    return out


def fit_single_model(train_part, seed, depth=6, learning_rate=0.03, iterations=800):
    data_train = skrub.var("data", train_part)
    data_train_fe = data_train.skb.apply_func(add_housing_ratio_features)
    X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train_fe[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    model = CatBoostRegressor(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=0,
        random_seed=seed,
        early_stopping_rounds=50,
    )

    pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred_graph.skb.make_learner(fitted=True)
    return learner


def build_split_and_fit(seed_split, seed_model):
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=seed_split
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()
    learner = fit_single_model(train_part, seed=seed_model)
    valid_pred = learner.predict({"data": valid_part})
    rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    return learner, valid_part, rmse


member_specs = [
    (42, 42),
    (7, 7),
    (123, 123),
]

learners = []
valid_parts = []
rmses = []

for split_seed, model_seed in member_specs:
    learner, valid_part, rmse = build_split_and_fit(split_seed, model_seed)
    learners.append(learner)
    valid_parts.append(valid_part)
    rmses.append(rmse)

base_train_idx, base_valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
base_train_part = train_df.iloc[base_train_idx].copy()
base_valid_part = train_df.iloc[base_valid_idx].copy()

base_learners = []
base_rmses = []
for split_seed, model_seed in member_specs:
    train_idx, _ = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=split_seed
    )
    train_part = train_df.iloc[train_idx].copy()
    learner = fit_single_model(train_part, seed=model_seed)
    base_learners.append(learner)
    pred = learner.predict({"data": base_valid_part})
    base_rmses.append(mean_squared_error(base_valid_part[target_col], pred) ** 0.5)

weights = np.array([1.0 / max(r, 1e-12) for r in base_rmses], dtype=float)
weights = weights / weights.sum()

ensemble_preds = None
for w, learner in zip(weights, base_learners):
    pred = learner.predict({"data": base_valid_part})
    if ensemble_preds is None:
        ensemble_preds = w * pred
    else:
        ensemble_preds += w * pred

final_validation_score = mean_squared_error(base_valid_part[target_col], ensemble_preds) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

def fit_full_model(train_full, seed):
    data_full = skrub.var("data", train_full)
    data_full_fe = data_full.skb.apply_func(add_housing_ratio_features)
    X_full = data_full_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_full = data_full_fe[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostRegressor(
        depth=6,
        learning_rate=0.03,
        iterations=800,
        loss_function="RMSE",
        eval_metric="RMSE",
        verbose=0,
        random_seed=seed,
        early_stopping_rounds=50,
    )

    pred_graph = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
    learner = pred_graph.skb.make_learner(fitted=True)
    return learner

full_learners = []
for _, model_seed in member_specs:
    full_learners.append(fit_full_model(train_df, seed=model_seed))

test_preds = None
for w, learner in zip(weights, full_learners):
    pred = learner.predict({"data": test_df})
    if test_preds is None:
        test_preds = w * pred
    else:
        test_preds += w * pred

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"median_house_value": test_preds})
submission.to_csv("./final/submission.csv", index=False)
