
import json
import numpy as np
import pandas as pd
import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

# Basic preprocessing helper
def fe_func(df):
    out = df.copy()
    if "Cabin" in out.columns:
        cabin = out["Cabin"].astype("string")
        cabin_split = cabin.str.split("/", expand=True)
        out["CabinDeck"] = cabin_split[0]
        out["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
        out["CabinSide"] = cabin_split[2]
        out = out.drop(columns=["Cabin"])
    if "Name" in out.columns:
        out["NameLength"] = out["Name"].astype("string").str.len()
    if "PassengerId" in out.columns:
        out["Group"] = out["PassengerId"].astype("string").str.split("_").str[0]
    for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
            out[f"{col}_log"] = np.log1p(out[col])
    return out

# Split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps bind on train_part only
data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(fe_func)

X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

# Shared preprocessing
vectorizer = skrub.TableVectorizer()

# Base models
lgbm_model = LGBMClassifier(
    n_estimators=160,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)

cat_model = CatBoostClassifier(
    iterations=160,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    random_seed=42,
    verbose=0,
)

log_model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
)

# Choice-based tuning block
lgbm_learning_rate = skrub.choose_float(0.01, 0.15, log=True, default=0.05, name="lgbm_learning_rate")
lgbm_num_leaves = skrub.choose_int(16, 64, default=31, name="lgbm_num_leaves")
cat_depth = skrub.choose_int(4, 8, default=6, name="cat_depth")
cat_learning_rate = skrub.choose_float(0.01, 0.15, log=True, default=0.05, name="cat_learning_rate")
log_c = skrub.choose_float(0.1, 10.0, log=True, default=1.0, name="log_c")

lgbm_tuned = LGBMClassifier(
    n_estimators=120,
    learning_rate=lgbm_learning_rate,
    num_leaves=lgbm_num_leaves,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=1,
    verbose=-1,
)

cat_tuned = CatBoostClassifier(
    iterations=120,
    learning_rate=cat_learning_rate,
    depth=cat_depth,
    loss_function="Logloss",
    random_seed=42,
    verbose=0,
)

log_tuned = LogisticRegression(
    max_iter=1000,
    C=log_c,
    solver="lbfgs",
)

voting_model = VotingClassifier(
    estimators=[
        ("lgbm", lgbm_tuned),
        ("cat", cat_tuned),
        ("log", log_tuned),
    ],
    voting="soft",
    weights=skrub.choose_from(
        {
            "equal": [1, 1, 1],
            "lgbm_heavy": [2, 1, 1],
            "cat_heavy": [1, 2, 1],
        },
        name="voting_weights",
    ),
)

pred = X_train.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(voting_model, y=y_train)

search = pred.skb.make_randomized_search(
    n_iter=4,
    n_jobs=1,
    random_state=42,
    fitted=True,
)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred)
if valid_pred.dtype != bool:
    valid_pred = valid_pred.astype(bool)

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

best_params = search.best_params_ if hasattr(search, "best_params_") else {}
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
