
import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import skrub
except Exception as e:
    raise ImportError(
        "This script requires skrub. Please install it before running."
    ) from e

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error


TARGET = "median_house_value"
INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "total_rooms" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    if "total_bedrooms" in out.columns and "total_rooms" in out.columns:
        denom = out["total_rooms"].replace(0, np.nan)
        out["bedrooms_per_room"] = (out["total_bedrooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    if "population" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["population_per_household"] = (out["population"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    if "population" in out.columns and "total_rooms" in out.columns:
        denom = out["total_rooms"].replace(0, np.nan)
        out["population_per_room"] = (out["population"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)

    if "longitude" in out.columns and "latitude" in out.columns:
        out["coord_radius"] = np.sqrt(out["longitude"] ** 2 + out["latitude"] ** 2)
        out["coord_product"] = out["longitude"] * out["latitude"]

    return out


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

data = skrub.var("data", train_df)
data_fe = data.skb.apply_func(skrub.deferred(add_ratio_features))

X = data_fe.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y = data_fe[TARGET].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=8,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    random_state=42,
)

pred = X.skb.apply(vectorizer).skb.apply(model, y=y)

# Holdout validation for a reproducible score print.
rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
split = int(len(idx) * 0.85)
fit_idx = idx[:split]
val_idx = idx[split:]

fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

fit_data = skrub.var("data", fit_df)
fit_data_fe = fit_data.skb.apply_func(skrub.deferred(add_ratio_features))
fit_X = fit_data_fe.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
fit_y = fit_data_fe[TARGET].skb.mark_as_y()
fit_pred = fit_X.skb.apply(vectorizer).skb.apply(model, y=fit_y)
learner = fit_pred.skb.make_learner(fitted=True)

val_pred = learner.predict({"data": val_df})
final_validation_score = mean_squared_error(val_df[TARGET].values, val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full training data and predict test set.
full_learner = pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({TARGET: test_pred})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
