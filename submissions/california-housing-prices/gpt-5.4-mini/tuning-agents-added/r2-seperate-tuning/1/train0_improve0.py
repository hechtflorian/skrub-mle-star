

import os
import pandas as pd
import numpy as np
import skrub
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

@skrub.deferred
def add_housing_features(df):
    out = df.copy()

    # Lightweight cleaning for known problematic numeric fields
    if "total_bedrooms" in out.columns:
        out["total_bedrooms"] = pd.to_numeric(out["total_bedrooms"], errors="coerce")

    for col in ["total_rooms", "total_bedrooms", "population", "households"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    households = out["households"].replace(0, np.nan) if "households" in out.columns else np.nan
    total_rooms = out["total_rooms"] if "total_rooms" in out.columns else np.nan
    total_bedrooms = out["total_bedrooms"] if "total_bedrooms" in out.columns else np.nan
    population = out["population"] if "population" in out.columns else np.nan

    if "total_rooms" in out.columns and "households" in out.columns:
        out["rooms_per_household"] = (total_rooms / households).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "total_bedrooms" in out.columns and "total_rooms" in out.columns:
        out["bedrooms_per_room"] = (total_bedrooms / total_rooms).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "population" in out.columns and "households" in out.columns:
        out["population_per_household"] = (population / households).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "total_bedrooms" in out.columns and "households" in out.columns:
        out["bedrooms_per_household"] = (total_bedrooms / households).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if "latitude" in out.columns and "longitude" in out.columns:
        out["lat_long_interaction"] = out["latitude"] * out["longitude"]

    return out

data = skrub.var("data", train_df)
data_fe = data.skb.apply_func(add_housing_features)

X = data_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
)

X_vec = X.skb.apply(vectorizer)

model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=96,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
)

predictor = X_vec.skb.apply(model, y=y)
learner = predictor.skb.make_learner(fitted=True)

# Real holdout validation split for the printed score
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
)

train_fold = train_df.iloc[train_idx].reset_index(drop=True)
val_fold = train_df.iloc[val_idx].reset_index(drop=True)

train_data = skrub.var("train_data", train_fold)
train_data_fe = train_data.skb.apply_func(add_housing_features)
X_train = train_data_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = train_data_fe[target_col].skb.mark_as_y()

X_train_vec = X_train.skb.apply(vectorizer)
holdout_predictor = X_train_vec.skb.apply(model, y=y_train)
holdout_learner = holdout_predictor.skb.make_learner(fitted=True)

val_preds = holdout_learner.predict({"train_data": val_fold})
rmse = mean_squared_error(val_fold[target_col], val_preds) ** 0.5

print(f"Final Validation Performance: {rmse}")

# Fit on full training data for submission
full_learner = learner
test_preds = full_learner.predict({"data": test_df})
submission = pd.DataFrame({target_col: test_preds})
submission.to_csv("submission.csv", index=False)
