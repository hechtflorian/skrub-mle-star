
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")

target_col = "Transported"
random_state = 42

train_df = pd.read_csv(TRAIN_PATH)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)


def engineer_passenger_features(df):
    out = df.copy()

    out = out.drop(columns=["PassengerId", "Name"], errors="ignore")

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype("string").str.split("/", n=2, expand=True)
        if cabin_parts is not None:
            if 0 in cabin_parts.columns:
                out["CabinDeck"] = cabin_parts[0]
            if 1 in cabin_parts.columns:
                out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
            if 2 in cabin_parts.columns:
                out["CabinSide"] = cabin_parts[2]
        out = out.drop(columns=["Cabin"], errors="ignore")

    return out


def to_bool_series(values, index=None):
    s = pd.Series(values, index=index)
    mapped = s.astype(str).map({"True": True, "False": False})
    return mapped.fillna(s).astype(bool)


def get_positive_scores(learner, df):
    try:
        proba = np.asarray(learner.predict_proba({"data": df}))
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)
    except Exception:
        pred = learner.predict({"data": df})
        return to_bool_series(pred).astype(float).to_numpy()


data_train_fe = data_train.skb.apply_func(engineer_passenger_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)
predictor_lgbm = (
    X_train
    .skb.apply(vectorizer_lgbm)
    .skb.apply(lgbm_model, y=y_train)
)
learner_lgbm = predictor_lgbm.skb.make_learner(fitted=True)

vectorizer_cat = skrub.TableVectorizer()
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    random_state=random_state,
    verbose=0,
)
predictor_cat = (
    X_train
    .skb.apply(vectorizer_cat)
    .skb.apply(cat_model, y=y_train)
)
learner_cat = predictor_cat.skb.make_learner(fitted=True)

valid_true = to_bool_series(valid_part[target_col], index=valid_part.index)
p_lgbm = get_positive_scores(learner_lgbm, valid_part)
p_cat = get_positive_scores(learner_cat, valid_part)

weight_grid = [
    (1.0, 0.0),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
]

best_score = -1.0
best_pred = None

for w_lgbm, w_cat in weight_grid:
    blended = w_lgbm * p_lgbm + w_cat * p_cat
    pred_bool = blended >= 0.5
    score = accuracy_score(valid_true, pred_bool)
    if score > best_score:
        best_score = score
        best_pred = pred_bool

final_validation_score = accuracy_score(valid_true, best_pred)
print(f"Final Validation Performance: {final_validation_score}")
