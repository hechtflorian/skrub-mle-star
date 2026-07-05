
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

def fe(df):
    df = df.copy()
    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype("string").fillna("")
        cabin_split = cabin.str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
    if "Name" in df.columns:
        name = df["Name"].astype("string").fillna("")
        df["NameLen"] = name.str.len()
    if "PassengerId" in df.columns:
        pid = df["PassengerId"].astype("string").fillna("")
        pid_split = pid.str.split("_", expand=True)
        if pid_split.shape[1] >= 2:
            df["GroupId"] = pid_split[0]
            df["GroupSizeHint"] = pd.to_numeric(pid_split[1], errors="coerce")
    return df

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(fe)

X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=0,
    random_seed=random_state,
)
lgbm_model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=random_state,
    verbose=-1,
)

vectorizer = skrub.TableVectorizer()

cat_pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
lgbm_pred = X_train.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_train)

cat_learner = cat_pred.skb.make_learner(fitted=True)
lgbm_learner = lgbm_pred.skb.make_learner(fitted=True)

def get_pred(learner, df):
    pred = learner.predict_proba({"data": df})
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:
        return pred[:, 1]
    return pred.ravel()

cat_valid_raw = get_pred(cat_learner, valid_part)
lgbm_valid_raw = get_pred(lgbm_learner, valid_part)

eps = 1e-6
cat_clip = np.clip(cat_valid_raw, eps, 1 - eps)
lgbm_clip = np.clip(lgbm_valid_raw, eps, 1 - eps)

def logit(p):
    return np.log(p / (1 - p))

avg_raw = 0.5 * (cat_valid_raw + lgbm_valid_raw)
abs_diff = np.abs(cat_valid_raw - lgbm_valid_raw)
max_raw = np.maximum(cat_valid_raw, lgbm_valid_raw)

meta_X = np.column_stack([
    logit(cat_clip),
    logit(lgbm_clip),
    avg_raw,
    abs_diff,
    max_raw,
])

meta_model = LogisticRegression(
    C=1.0,
    solver="lbfgs",
    max_iter=1000,
    random_state=random_state,
)
meta_model.fit(meta_X, valid_part[target_col].astype(int).values)

valid_meta_prob = meta_model.predict_proba(meta_X)[:, 1]
valid_pred = valid_meta_prob >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
data_full = data_full.skb.apply_func(fe)

X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_cat_pred = X_full.skb.apply(vectorizer).skb.apply(cat_model, y=y_full)
full_lgbm_pred = X_full.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_full)

full_cat_learner = full_cat_pred.skb.make_learner(fitted=True)
full_lgbm_learner = full_lgbm_pred.skb.make_learner(fitted=True)

test_cat_raw = get_pred(full_cat_learner, test_df)
test_lgbm_raw = get_pred(full_lgbm_learner, test_df)

test_cat_clip = np.clip(test_cat_raw, eps, 1 - eps)
test_lgbm_clip = np.clip(test_lgbm_raw, eps, 1 - eps)

test_avg_raw = 0.5 * (test_cat_raw + test_lgbm_raw)
test_abs_diff = np.abs(test_cat_raw - test_lgbm_raw)
test_max_raw = np.maximum(test_cat_raw, test_lgbm_raw)

test_meta_X = np.column_stack([
    logit(test_cat_clip),
    logit(test_lgbm_clip),
    test_avg_raw,
    test_abs_diff,
    test_max_raw,
])

test_meta_prob = meta_model.predict_proba(test_meta_X)[:, 1]
test_pred = test_meta_prob >= 0.5

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    target_col: test_pred.astype(bool),
})
submission.to_csv("./final/submission.csv", index=False)
