
import os
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

def add_features(df):
    out = df.copy()
    out["CabinDeck"] = out["Cabin"].fillna("X").astype(str).str.split("/").str[0]
    out["CabinNum"] = pd.to_numeric(out["Cabin"].fillna("X").astype(str).str.split("/").str[1], errors="coerce")
    out["CabinSide"] = out["Cabin"].fillna("X").astype(str).str.split("/").str[2]
    out["Group"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["GroupSize"] = out["Group"].map(out.groupby("Group").size())
    out["Surname"] = out["Name"].fillna("Unknown").astype(str).str.split(" ").str[-1]
    out["TotalSpend"] = out[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    out["IsAlone"] = (out["GroupSize"] == 1).astype(int)
    out["AgeGroup"] = pd.cut(out["Age"], bins=[-1, 12, 18, 25, 35, 50, 65, 200], labels=False)
    out["SpendPerAge"] = (out["TotalSpend"] / out["Age"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["CabinKnown"] = (out["Cabin"].notna()).astype(int)
    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_features)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

model_lgbm = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

model_cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.08,
    loss_function="Logloss",
    random_seed=random_state,
    verbose=0,
)

pred_lgbm = X_train.skb.apply(vectorizer_lgbm).skb.apply(model_lgbm, y=y_train)
pred_cat = X_train.skb.apply(vectorizer_cat).skb.apply(model_cat, y=y_train)

learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_cat.skb.make_learner(fitted=True)

def _get_proba(learner, df):
    p = learner.predict_proba({"data": df})
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] > 1:
        return p[:, 1].astype(float)
    pred = learner.predict({"data": df})
    pred = np.asarray(pred)
    if pred.dtype == bool:
        return pred.astype(float)
    if np.issubdtype(pred.dtype, np.floating):
        return pred.astype(float).ravel()
    return pd.Series(pred).astype(str).str.lower().isin(["true", "1", "yes"]).astype(float).to_numpy()

train_pred_lgbm = _get_proba(learner_lgbm, train_part)
train_pred_cat = _get_proba(learner_cat, train_part)
valid_pred_lgbm = _get_proba(learner_lgbm, valid_part)
valid_pred_cat = _get_proba(learner_cat, valid_part)

meta_X_train = np.column_stack([train_pred_lgbm, train_pred_cat])
meta_X_valid = np.column_stack([valid_pred_lgbm, valid_pred_cat])

from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression(random_state=random_state, max_iter=1000)
meta_model.fit(meta_X_train, train_part[target_col].astype(int).to_numpy())

valid_meta_prob = meta_model.predict_proba(meta_X_valid)[:, 1]
blend_pred = valid_meta_prob >= 0.5

final_validation_score = accuracy_score(valid_part[target_col].astype(int), blend_pred.astype(int))
print(f"Final Validation Performance: {final_validation_score}")

test_df = pd.read_csv(os.path.join("./input", "test.csv"))


import os
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

def add_features(df):
    out = df.copy()
    out["CabinDeck"] = out["Cabin"].fillna("X").astype(str).str.split("/").str[0]
    out["CabinNum"] = pd.to_numeric(out["Cabin"].fillna("X").astype(str).str.split("/").str[1], errors="coerce")
    out["CabinSide"] = out["Cabin"].fillna("X").astype(str).str.split("/").str[2]
    out["Group"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["GroupSize"] = out["Group"].map(out.groupby("Group").size())
    out["Surname"] = out["Name"].fillna("Unknown").astype(str).str.split(" ").str[-1]
    out["TotalSpend"] = out[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    out["IsAlone"] = (out["GroupSize"] == 1).astype(int)
    out["AgeGroup"] = pd.cut(out["Age"], bins=[-1, 12, 18, 25, 35, 50, 65, 200], labels=False)
    out["SpendPerAge"] = (out["TotalSpend"] / out["Age"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["CabinKnown"] = (out["Cabin"].notna()).astype(int)
    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# Block 1: honest holdout validation fit on train_part only
data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_features)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

model_lgbm = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

model_cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.08,
    loss_function="Logloss",
    random_seed=random_state,
    verbose=0,
)

pred_lgbm = X_train.skb.apply(vectorizer_lgbm).skb.apply(model_lgbm, y=y_train)
pred_cat = X_train.skb.apply(vectorizer_cat).skb.apply(model_cat, y=y_train)

learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_cat.skb.make_learner(fitted=True)

def _get_proba(learner, df):
    p = learner.predict_proba({"data": df})
    p = np.asarray(p)
    if p.ndim == 2 and p.shape[1] > 1:
        return p[:, 1].astype(float)
    pred = learner.predict({"data": df})
    pred = np.asarray(pred)
    if pred.dtype == bool:
        return pred.astype(float)
    if np.issubdtype(pred.dtype, np.floating):
        return pred.astype(float).ravel()
    return pd.Series(pred).astype(str).str.lower().isin(["true", "1", "yes"]).astype(float).to_numpy()

train_pred_lgbm = _get_proba(learner_lgbm, train_part)
train_pred_cat = _get_proba(learner_cat, train_part)
valid_pred_lgbm = _get_proba(learner_lgbm, valid_part)
valid_pred_cat = _get_proba(learner_cat, valid_part)

meta_X_train = np.column_stack([train_pred_lgbm, train_pred_cat])
meta_X_valid = np.column_stack([valid_pred_lgbm, valid_pred_cat])

from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression(random_state=random_state, max_iter=1000)
meta_model.fit(meta_X_train, train_part[target_col].astype(int).to_numpy())

valid_meta_prob = meta_model.predict_proba(meta_X_valid)[:, 1]
blend_pred = valid_meta_prob >= 0.5

final_validation_score = accuracy_score(valid_part[target_col].astype(int), blend_pred.astype(int))
print(f"Final Validation Performance: {final_validation_score}")

# Block 2: submission-stage full-train refit and test prediction
test_df = pd.read_csv(os.path.join("./input", "test.csv"))

data_full = skrub.var("data", train_df)
data_full = data_full.skb.apply_func(add_features)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

vectorizer_lgbm_full = skrub.TableVectorizer()
vectorizer_cat_full = skrub.TableVectorizer()

pred_lgbm_full = X_full.skb.apply(vectorizer_lgbm_full).skb.apply(model_lgbm, y=y_full)
pred_cat_full = X_full.skb.apply(vectorizer_cat_full).skb.apply(model_cat, y=y_full)

learner_lgbm_full = pred_lgbm_full.skb.make_learner(fitted=True)
learner_cat_full = pred_cat_full.skb.make_learner(fitted=True)

test_pred_lgbm = _get_proba(learner_lgbm_full, test_df)
test_pred_cat = _get_proba(learner_cat_full, test_df)

test_meta_X = np.column_stack([test_pred_lgbm, test_pred_cat])
test_meta_prob = meta_model.predict_proba(test_meta_X)[:, 1]
test_pred = test_meta_prob >= 0.5

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    target_col: pd.Series(test_pred).map({True: "True", False: "False"}).to_numpy(),
})
submission.to_csv("./final/submission.csv", index=False)


os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    target_col: pd.Series(test_pred).map({True: "True", False: "False"}).to_numpy(),
})
submission.to_csv("./final/submission.csv", index=False)
