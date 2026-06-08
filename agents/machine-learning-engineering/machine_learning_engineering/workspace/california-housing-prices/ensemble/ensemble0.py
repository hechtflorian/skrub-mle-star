
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
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Keep a DataOps-first flow: define data, mark X/y, and use .skb.apply(...)
data = skrub.var("data", train_df)

X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

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

X_clean = X.skb.apply_func(fill_missing_numeric)
X_feat = X_clean.skb.apply_func(add_ratio_features)

# Baked best params from terminal tuning
def make_model(seed):
    return HistGradientBoostingRegressor(
        learning_rate=0.05993292420985183,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=13,
        l2_regularization=0.0,
        random_state=seed,
    )

def fit_predict(seed, sample_idx=None):
    if sample_idx is None:
        train_part = train_df.copy()
    else:
        train_part = train_df.iloc[sample_idx].copy()

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_part)), test_size=0.2, random_state=seed
    )

    train_split = train_part.iloc[train_idx].copy()
    valid_split = train_part.iloc[valid_idx].copy()

    data_split = skrub.var("data", train_split)
    X_split = data_split.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y_split = data_split[TARGET].skb.mark_as_y()

    X_split_clean = X_split.skb.apply_func(fill_missing_numeric)
    X_split_feat = X_split_clean.skb.apply_func(add_ratio_features)

    model = make_model(seed)

    pred = X_split_feat.skb.apply(model, y=y_split)
    learner = pred.skb.make_learner(fitted=True)

    valid_pred = learner.predict({"data": valid_split})
    valid_score = mean_squared_error(valid_split[TARGET], valid_pred) ** 0.5

    # Fit on the full variant data for test prediction
    data_full = skrub.var("data", train_part)
    X_full = data_full.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y_full = data_full[TARGET].skb.mark_as_y()

    X_full_clean = X_full.skb.apply_func(fill_missing_numeric)
    X_full_feat = X_full_clean.skb.apply_func(add_ratio_features)

    full_pred = X_full_feat.skb.apply(model, y=y_full)
    full_learner = full_pred.skb.make_learner(fitted=True)
    test_pred = full_learner.predict({"data": test_df})

    return valid_score, test_pred

# Minimal diversity: 3 seeds, one bootstrap variant
seeds = [42, 7, 2024]
all_valid_scores = []
all_test_preds = []

# Variant 1: standard split with seed 42
score, pred = fit_predict(seeds[0])
all_valid_scores.append(score)
all_test_preds.append(pred)

# Variant 2: standard split with seed 7
score, pred = fit_predict(seeds[1])
all_valid_scores.append(score)
all_test_preds.append(pred)

# Variant 3: bootstrap resample with seed 2024
rng = np.random.default_rng(seeds[2])
bootstrap_idx = rng.integers(0, len(train_df), size=len(train_df))
score, pred = fit_predict(seeds[2], sample_idx=bootstrap_idx)
all_valid_scores.append(score)
all_test_preds.append(pred)

# Uniform blend of all test predictions
test_pred = np.mean(np.column_stack(all_test_preds), axis=1)

# Validation metric reported as average hold-out RMSE across variants
final_validation_score = float(np.mean(all_valid_scores))
print(f"Final Validation Performance: {final_validation_score}")

# Keep the original submission path unchanged
submission = pd.DataFrame({TARGET: test_pred})
submission.to_csv("submission.csv", index=False)
print(submission.head().to_string(index=False))
