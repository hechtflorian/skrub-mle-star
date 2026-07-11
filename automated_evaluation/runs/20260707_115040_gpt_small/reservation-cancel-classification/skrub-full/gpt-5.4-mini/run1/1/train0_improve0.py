
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_booking_history_interactions(df):
    out = df.copy()
    cols = set(out.columns)

    # Guarded ratio / interaction features for related booking-history columns.
    # Keep this narrowly focused on clearly related counts / binary repeated_guest fields.
    feature_specs = [
        ("repeated_guest", "previous_cancellations", "repeated_guest_x_previous_cancellations", "ratio"),
        ("repeated_guest", "previous_bookings_not_canceled", "repeated_guest_x_previous_bookings_not_canceled", "ratio"),
        ("previous_cancellations", "previous_bookings_not_canceled", "cancellation_rate_history", "ratio"),
        ("booking_changes", "previous_bookings_not_canceled", "booking_changes_per_previous_booking", "ratio"),
        ("booking_changes", "previous_cancellations", "booking_changes_per_previous_cancellation", "ratio"),
    ]

    for numer_col, denom_col, out_name, mode in feature_specs:
        if numer_col in cols and denom_col in cols:
            numer = pd.to_numeric(out[numer_col], errors="coerce")
            denom = pd.to_numeric(out[denom_col], errors="coerce")
            if mode == "ratio":
                denom_safe = denom.replace(0, np.nan)
                out[out_name] = (numer / denom_safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            else:
                out[out_name] = (numer * denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def build_lgb_graph(data_train):
    data_fe = data_train.skb.apply_func(add_booking_history_interactions)
    X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_fe[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    # Keep the stronger LGBM branch as the main path; preserve fixed parameters.
    pred_graph_lgb = X.skb.apply(vectorizer).skb.apply(model_lgb, y=y)
    return pred_graph_lgb


data_train = skrub.var("data", train_part)
pred_graph_lgb = build_lgb_graph(data_train)
learner_lgb = pred_graph_lgb.skb.make_learner(fitted=True)

# Keep CatBoost branch fixed as a reference, but do not ensemble equally anymore.
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

data_train_cat = skrub.var("data", train_part)
data_fe_cat = data_train_cat.skb.apply_func(add_booking_history_interactions)
X_cat = data_fe_cat.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_cat = data_fe_cat[target_col].skb.mark_as_y()
pred_graph_cat = X_cat.skb.apply(skrub.TableVectorizer()).skb.apply(model_cat, y=y_cat)
learner_cat = pred_graph_cat.skb.make_learner(fitted=True)

valid_pred_lgb = np.asarray(learner_lgb.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part}), dtype=float).ravel()

# Structural refinement plan: prioritize the stronger LGBM branch.
valid_pred = valid_pred_lgb
final_validation_score = roc_auc_score(valid_part[target_col], valid_pred)

print(f"Final Validation Performance: {final_validation_score}")
