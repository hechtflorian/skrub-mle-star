

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


@skrub.deferred
def add_geo_features(df):
    out = df.copy()
    cols = set(out.columns)

    if {"longitude", "latitude"}.issubset(cols):
        lon = out["longitude"]
        lat = out["latitude"]

        out["longitude_abs"] = lon.abs()
        out["latitude_abs"] = lat.abs()
        out["lon_lat_interaction"] = lon * lat
        out["coord_radius"] = np.sqrt(lon ** 2 + lat ** 2)
        out["coord_manhattan"] = lon.abs() + lat.abs()
        out["coord_balance"] = (lon.abs() - lat.abs()).abs()

        # Drop one redundant member of the extreme collinearity cluster only when
        # the engineered geo block is present.
        out = out.drop(columns=["longitude"], errors="ignore")

    return out


train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_geo_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
predictor = X_train.skb.apply(vectorizer).skb.apply(
    CatBoostRegressor(verbose=0, random_seed=42),
    y=y_train,
)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
