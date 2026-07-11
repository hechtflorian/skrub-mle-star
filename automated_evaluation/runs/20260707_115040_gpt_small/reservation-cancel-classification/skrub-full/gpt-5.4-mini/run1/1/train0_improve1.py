
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import skrub
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

random_state = 42
target_col = "booking_status"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data = skrub.var("data", train_part)
X_train = data.drop(columns=[target_col, "id"], errors="ignore").skb.mark_as_X()
y_train = data[target_col].skb.mark_as_y()

# Model 1: LightGBM baseline from base solution
vectorizer = skrub.TableVectorizer()
model_lgb = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)


import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_guarded_booking_features(df):
    out = df.copy()

    # Guarded interaction features focused on booking-history signal
    if "repeated_guest" in out.columns:
        repeated = pd.to_numeric(out["repeated_guest"], errors="coerce").fillna(0.0)
        if "previous_bookings_not_canceled" in out.columns:
            prev_nc = pd.to_numeric(out["previous_bookings_not_canceled"], errors="coerce").fillna(0.0)
            out["repeated_guest_prev_bookings_nc"] = repeated * prev_nc
            denom = (prev_nc + 1.0).replace(0, np.nan)
            out["repeated_guest_prev_bookings_nc_ratio"] = (
                (repeated / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )
        if "previous_cancellations" in out.columns:
            prev_can = pd.to_numeric(out["previous_cancellations"], errors="coerce").fillna(0.0)
            out["repeated_guest_prev_cancellations"] = repeated * prev_can
            denom = (prev_can + 1.0).replace(0, np.nan)
            out["repeated_guest_prev_cancellations_ratio"] = (
                (repeated / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )

    # Stable rate features with minimal redundancy
    if "adr" in out.columns and "lead_time" in out.columns:
        adr = pd.to_numeric(out["adr"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        lead_time = pd.to_numeric(out["lead_time"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["adr_per_lead_time"] = (adr / (lead_time + 1.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "adr" in out.columns and "stays_in_weekend_nights" in out.columns:
        adr = pd.to_numeric(out["adr"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        weekend = pd.to_numeric(out["stays_in_weekend_nights"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out["adr_per_weekend_night"] = (adr / (weekend + 1.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Redundancy cleanup: drop id and the raw count if the engineered ratios capture it
    drop_cols = []
    if "id" in out.columns:
        drop_cols.append("id")
    if "previous_bookings_not_canceled" in out.columns and "repeated_guest_prev_bookings_nc_ratio" in out.columns:
        drop_cols.append("previous_bookings_not_canceled")
    if "previous_cancellations" in out.columns and "repeated_guest_prev_cancellations_ratio" in out.columns:
        drop_cols.append("previous_cancellations")
    out = out.drop(columns=drop_cols, errors="ignore")

    return out


data_train = skrub.var("data", train_part)
data_fe = data_train.skb.apply_func(add_guarded_booking_features)

X_train = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

# Stronger LGBM path kept as the main validation path
model_lgb = lgb_model if "lgb_model" in globals() else model_lgb
pred_graph_lgb = X_train.skb.apply(vectorizer).skb.apply(model_lgb, y=y_train)
learner_lgb = pred_graph_lgb.skb.make_learner(fitted=True)
valid_pred_lgb = np.asarray(learner_lgb.predict({"data": valid_part}), dtype=float).ravel()

# CatBoost kept only as a comparison branch, not blended into the final score
model_cat = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=random_state,
    verbose=0,
    allow_writing_files=False,
)
pred_graph_cat = X_train.skb.apply(vectorizer).skb.apply(model_cat, y=y_train)
learner_cat = pred_graph_cat.skb.make_learner(fitted=True)
valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part}), dtype=float).ravel()

valid_pred = valid_pred_lgb
final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)

print(f"Final Validation Performance: {final_validation_score}")
