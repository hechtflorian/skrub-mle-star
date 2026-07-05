
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

input_dir = "./input"
train_df = pd.read_csv(os.path.join(input_dir, "train.csv"))

def feature_engineer(df):
    df = df.copy()
    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype(str).str.split("_", n=1, expand=True)
        df["PassengerGroup"] = pd.to_numeric(pid[0], errors="coerce")
        df["PassengerNumber"] = pd.to_numeric(pid[1], errors="coerce")
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype(str).str.split("/", n=2, expand=True)
        df["CabinDeck"] = cabin[0]
        df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
        df["CabinSide"] = cabin[2]
    if "Name" in df.columns:
        df["Surname"] = df["Name"].astype(str).str.split(" ", n=1).str[-1]
        df["NameLen"] = df["Name"].astype(str).str.len()
    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpending"] = df[spending_cols].fillna(0).sum(axis=1)
    df["SpendingFlag"] = (df["TotalSpending"] > 0).astype(int)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-np.inf, 12, 18, 30, 50, np.inf],
        labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
    )
    return df

def get_positive_score(learner, df):
    try:
        proba = learner.predict_proba({"data": df})
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].ravel()
        return proba.ravel()
    except Exception:
        pred = learner.predict({"data": df})
        pred = np.asarray(pred)
        if pred.dtype.kind in "biu":
            return pred.astype(float).ravel()
        return pred.ravel()

train_part, valid_part = train_test_split(
    train_df, test_size=0.2, random_state=random_state, stratify=train_df[target_col]
)

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(feature_engineer)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

pred_lgbm = X_train.skb.apply(
    vectorizer,
).skb.apply(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    ),
    y=y_train,
)

pred_cat = X_train.skb.apply(
    vectorizer,
).skb.apply(
    CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        random_seed=random_state,
        verbose=0,
    ),
    y=y_train,
)

learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_cat.skb.make_learner(fitted=True)

valid_pred_lgbm = get_positive_score(learner_lgbm, valid_part)
valid_pred_cat = get_positive_score(learner_cat, valid_part)

if valid_pred_lgbm.dtype.kind not in "fc":
    valid_pred_lgbm = valid_pred_lgbm.astype(float)
if valid_pred_cat.dtype.kind not in "fc":
    valid_pred_cat = valid_pred_cat.astype(float)

# Candidate blend rules selected on the holdout only
candidates = []

# 1) Linear soft blend over a small 2D simplex grid
grid = np.arange(0.0, 1.0001, 0.1)
best_linear_score = -1.0
best_linear_pred = None
best_linear_weights = (0.5, 0.5)

for w_lgbm in grid:
    w_cat = 1.0 - w_lgbm
    blended = w_lgbm * valid_pred_lgbm + w_cat * valid_pred_cat
    pred = blended >= 0.5
    score = accuracy_score(valid_part[target_col], pred)
    if score > best_linear_score:
        best_linear_score = score
        best_linear_pred = pred
        best_linear_weights = (w_lgbm, w_cat)

candidates.append(
    ("linear_soft_blend", best_linear_score, best_linear_pred, best_linear_weights)
)

# 2) Rank-average blend
rank_avg = (
    pd.Series(valid_pred_lgbm).rank(method="average", pct=True).to_numpy()
    + pd.Series(valid_pred_cat).rank(method="average", pct=True).to_numpy()
) / 2.0
rank_avg_pred = rank_avg >= 0.5
rank_avg_score = accuracy_score(valid_part[target_col], rank_avg_pred)
candidates.append(("rank_average_blend", rank_avg_score, rank_avg_pred, None))

# 3) Geometric mean blend
geom = np.sqrt(np.clip(valid_pred_lgbm, 1e-12, 1.0) * np.clip(valid_pred_cat, 1e-12, 1.0))
geom_pred = geom >= 0.5
geom_score = accuracy_score(valid_part[target_col], geom_pred)
candidates.append(("geometric_mean_blend", geom_score, geom_pred, None))

# 4) Majority-vote fallback from thresholded probs
vote_pred = ((valid_pred_lgbm >= 0.5).astype(int) + (valid_pred_cat >= 0.5).astype(int)) >= 1
vote_score = accuracy_score(valid_part[target_col], vote_pred)
candidates.append(("majority_vote_fallback", vote_score, vote_pred, None))

best_name, final_validation_score, final_pred, best_weights = max(
    candidates, key=lambda x: x[1]
)

print(f"Final Validation Performance: {final_validation_score}")
