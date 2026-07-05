
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def fe_func(df):
    out = df.copy()

    if "Cabin" in out.columns:
        cabin = out["Cabin"].astype("string")
        cabin_parts = cabin.str.split("/", expand=True)
        if cabin_parts.shape[1] >= 3:
            out["CabinDeck"] = cabin_parts[0].fillna("Unknown").astype("string")
            out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
            out["CabinSide"] = cabin_parts[2].fillna("Unknown").astype("string")

    if "Name" in out.columns:
        name = out["Name"].astype("string")
        out["NameLength"] = name.fillna("").str.len().astype("float")

    if "Age" in out.columns:
        age_bins = [-np.inf, 12, 18, 30, 50, np.inf]
        age_labels = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]
        out["AgeGroup"] = pd.cut(
            pd.to_numeric(out["Age"], errors="coerce"),
            bins=age_bins,
            labels=age_labels,
            include_lowest=True,
        ).astype("string")

    numeric_cols = [
        c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in out.columns
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if numeric_cols:
        out["TotalSpend"] = out[numeric_cols].sum(axis=1, min_count=1)
        out["HasSpend"] = (out["TotalSpend"].fillna(0) > 0).astype("int64")

    return out

def add_leg2_view(df):
    out = df.copy()

    if "TotalSpend" in out.columns:
        spend = pd.to_numeric(out["TotalSpend"], errors="coerce")
        bins = [-np.inf, 0, 1, 100, 1000, np.inf]
        labels = ["Zero", "Tiny", "Small", "Medium", "Large"]
        out["SpendBucket"] = pd.cut(spend, bins=bins, labels=labels, include_lowest=True).astype("string")

    if "CabinDeck" in out.columns and "CabinSide" in out.columns:
        deck = out["CabinDeck"].astype("string").fillna("Unknown")
        side = out["CabinSide"].astype("string").fillna("Unknown")
        out["CabinDeckSide"] = (deck + "_" + side).astype("string")

    return out

def predict_scores(learner, df):
    try:
        proba = learner.predict_proba({"data": df})
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)
    except Exception:
        pred = learner.predict({"data": df})
        pred = np.asarray(pred)
        if pred.dtype == bool:
            return pred.astype(float)
        return pred.astype(float).ravel()

def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float).ravel()
    if scores.size == 0:
        return scores
    smin = np.nanmin(scores)
    smax = np.nanmax(scores)
    if not np.isfinite(smin) or not np.isfinite(smax) or np.isclose(smin, smax):
        return np.clip(scores, 0.0, 1.0)
    return (scores - smin) / (smax - smin)

data_train = skrub.var("data", train_part)
data_fe = data_train.skb.apply_func(fe_func)

X_train = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_fe[target_col].skb.mark_as_y()

# Leg 1: original chain
vectorizer_1 = skrub.TableVectorizer()
pred_1 = X_train.skb.apply(vectorizer_1).skb.apply(
    HistGradientBoostingClassifier(random_state=random_state, max_depth=6, learning_rate=0.08),
    y=y_train,
)
learner_1 = pred_1.skb.make_learner(fitted=True)

# Leg 2: minimally augmented feature-view chain
X_train_leg2 = X_train.skb.apply_func(add_leg2_view)
vectorizer_2 = skrub.TableVectorizer()
pred_2 = X_train_leg2.skb.apply(vectorizer_2).skb.apply(
    HistGradientBoostingClassifier(random_state=random_state, max_depth=6, learning_rate=0.08),
    y=y_train,
)
learner_2 = pred_2.skb.make_learner(fitted=True)

# Validation predictions
valid_p1 = predict_scores(learner_1, valid_part)
valid_p2 = predict_scores(learner_2, valid_part)

# Rank/normalized blend for robustness
valid_p1 = normalize_scores(valid_p1)
valid_p2 = normalize_scores(valid_p2)
valid_blend = 0.65 * valid_p1 + 0.35 * valid_p2

# Small deterministic threshold tuning
threshold_grid = [0.45, 0.5, 0.55]
best_threshold = 0.5
best_score = -1.0

y_true = valid_part[target_col].astype(bool).to_numpy()
for thr in threshold_grid:
    valid_pred = valid_blend >= thr
    score = accuracy_score(y_true, valid_pred)
    if score > best_score:
        best_score = score
        best_threshold = thr

final_validation_score = best_score
print(f"Final Validation Performance: {final_validation_score}")
