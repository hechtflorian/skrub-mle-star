
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

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def prep_columns(df):
    out = df.copy()

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype("string").str.split("/", expand=True)
        out["CabinDeck"] = cabin_parts[0]
        out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        out["CabinSide"] = cabin_parts[2]
        out = out.drop(columns=["Cabin"])

    if "Name" in out.columns:
        out["NameLength"] = out["Name"].astype("string").str.len()
        out["NameTokens"] = out["Name"].astype("string").str.split().str.len()
        out = out.drop(columns=["Name"])

    bool_cols = [c for c in ["CryoSleep", "VIP"] if c in out.columns]
    for c in bool_cols:
        out[c] = out[c].astype("string").map({"True": 1, "False": 0}).astype(float)

    if "PassengerId" in out.columns:
        pid = out["PassengerId"].astype("string").str.split("_", expand=True)
        out["Group"] = pid[0]
        out["Person"] = pid[1]

    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Leg 1: LightGBM on engineered features
pred_graph_lgbm = X_train.skb.apply_func(prep_columns).skb.apply(
    skrub.TableVectorizer()
).skb.apply(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    ),
    y=y_train,
)

# Leg 2: CatBoost on same engineered features
pred_graph_cat = X_train.skb.apply_func(prep_columns).skb.apply(
    skrub.TableVectorizer()
).skb.apply(
    CatBoostClassifier(
        verbose=0,
        random_seed=random_state,
        loss_function="Logloss",
        allow_writing_files=False,
    ),
    y=y_train,
)

learner_lgbm = pred_graph_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_graph_cat.skb.make_learner(fitted=True)

def predict_proba_like(learner, df):
    pred = learner.predict({"data": df})
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:
        pred = pred[:, 1]
    return pred.astype(float).ravel()

valid_pred_lgbm = predict_proba_like(learner_lgbm, valid_part)
valid_pred_cat = predict_proba_like(learner_cat, valid_part)

valid_blend = 0.5 * valid_pred_lgbm + 0.5 * valid_pred_cat
valid_pred = valid_blend >= 0.5
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_graph_lgbm = X_full.skb.apply_func(prep_columns).skb.apply(
    skrub.TableVectorizer()
).skb.apply(
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    ),
    y=y_full,
)

full_pred_graph_cat = X_full.skb.apply_func(prep_columns).skb.apply(
    skrub.TableVectorizer()
).skb.apply(
    CatBoostClassifier(
        verbose=0,
        random_seed=random_state,
        loss_function="Logloss",
        allow_writing_files=False,
    ),
    y=y_full,
)

full_learner_lgbm = full_pred_graph_lgbm.skb.make_learner(fitted=True)
full_learner_cat = full_pred_graph_cat.skb.make_learner(fitted=True)

test_pred_lgbm = predict_proba_like(full_learner_lgbm, test_df)
test_pred_cat = predict_proba_like(full_learner_cat, test_df)

test_blend = 0.5 * test_pred_lgbm + 0.5 * test_pred_cat
test_pred = test_blend >= 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": pd.Series(test_pred).map({True: "True", False: "False"}).values,
    }
)
submission.to_csv("submission.csv", index=False)
