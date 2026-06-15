
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
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


def build_predictor(train_frame, seed, depth=None, learning_rate=None):
    data_train = skrub.var("data", train_frame)
    data_train = data_train.skb.apply_func(add_ratio_features)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    cat_params = dict(verbose=0, random_seed=seed)
    if depth is not None:
        cat_params["depth"] = depth
    if learning_rate is not None:
        cat_params["learning_rate"] = learning_rate

    predictor = X_train.skb.apply(vectorizer).skb.apply(
        CatBoostRegressor(**cat_params),
        y=y_train,
    )
    return predictor


# Branch A: original configuration
predictor_a = build_predictor(train_part, seed=42)
val_learner_a = predictor_a.skb.make_learner(fitted=True)
valid_pred_a = val_learner_a.predict({"data": valid_part})

# Branch B: minimal stochastic variation in the terminal learner
predictor_b = build_predictor(train_part, seed=123, depth=6)
val_learner_b = predictor_b.skb.make_learner(fitted=True)
valid_pred_b = val_learner_b.predict({"data": valid_part})

# Light blend of validation predictions
final_valid_pred = 0.5 * valid_pred_a + 0.5 * valid_pred_b
final_validation_score = mean_squared_error(valid_part[target_col], final_valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Refit both branches on the full training data for test-time blending
full_predictor_a = build_predictor(train_df, seed=42)
full_learner_a = full_predictor_a.skb.make_learner(fitted=True)
test_pred_a = full_learner_a.predict({"data": test_df})

full_predictor_b = build_predictor(train_df, seed=123, depth=6)
full_learner_b = full_predictor_b.skb.make_learner(fitted=True)
test_pred_b = full_learner_b.predict({"data": test_df})

pred_final = 0.5 * test_pred_a + 0.5 * test_pred_b
