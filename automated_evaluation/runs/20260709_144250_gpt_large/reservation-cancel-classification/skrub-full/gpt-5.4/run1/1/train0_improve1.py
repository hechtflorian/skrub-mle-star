
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "booking_status"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()


import numpy as np
import pandas as pd
import skrub
from skrub import DropCols

def add_booking_interactions(df):
    out = df.copy()

    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(out.columns):
        out["total_nights"] = (
            out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
        )

    guest_cols = [c for c in ["adults", "children", "babies"] if c in out.columns]
    if guest_cols:
        out["total_guests"] = out[guest_cols].sum(axis=1)

    if "lead_time" in out.columns and "total_guests" in out.columns:
        guest_denom = out["total_guests"].replace(0, np.nan)
        out["lead_time_per_guest"] = (
            (out["lead_time"] / guest_denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    if "lead_time" in out.columns and "total_nights" in out.columns:
        night_denom = out["total_nights"].replace(0, np.nan)
        out["lead_time_per_night"] = (
            (out["lead_time"] / night_denom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    return out

X_train_fe = X_train.skb.apply_func(add_booking_interactions).skb.apply(
    DropCols(cols=["id"])
)

model = CatBoostClassifier(
    random_state=random_state,
    verbose=0,
)

predictor = X_train_fe.skb.apply(model, y=y_train)


val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict_proba({"data": valid_part})[:, 1]
final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)

print(f"Final Validation Performance: {final_validation_score}")
