
import json
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_col = "Transported"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cabin = out["Cabin"].fillna("U/U/U").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["Spending"] = out[spending_cols].fillna(0).sum(axis=1)
    out["HasSpending"] = (out["Spending"] > 0).astype(int)

    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 30, 50, 120],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype("object")

    out["PassengerGroup"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["NameLength"] = out["Name"].fillna("").astype(str).str.len()
    out["Surname"] = out["Name"].fillna("").astype(str).str.split().str[-1]

    out["LuxurySpending"] = out[["Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["BasicSpending"] = out[["RoomService", "FoodCourt", "ShoppingMall"]].fillna(0).sum(axis=1)
    out["NoSpending"] = (out["Spending"] == 0).astype(int)

    out["FamilyNameKey"] = (
        out["Surname"].fillna("Unknown").astype(str)
        + "_"
        + out["PassengerGroup"].fillna("Unknown").astype(str)
    )

    return out


train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df[target_col].astype(int),
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)

X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].astype(int).skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgbm_model = LGBMClassifier(
    n_estimators=750,
    learning_rate=skrub.choose_float(0.02, 0.05, log=True, default=0.03, name="lgbm_learning_rate"),
    num_leaves=skrub.choose_int(24, 40, n_steps=9, default=31, name="lgbm_num_leaves"),
    random_state=42,
    verbose=-1,
    n_jobs=1,
)

cat_model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=0,
    random_seed=42,
    allow_writing_files=False,
    thread_count=1,
)

lgbm_graph = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)
cat_graph = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

pred = lgbm_graph.skb.apply_func(lambda x: x)  # keep graph object available
pred = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)
search = pred.skb.make_randomized_search(n_iter=4, n_jobs=1, random_state=42, fitted=True)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part}).astype(float)
final_validation_score = accuracy_score(valid_part[target_col].astype(int), (valid_pred >= 0.5).astype(int))
print(f"Final Validation Performance: {final_validation_score}")

best_params = {}
sp = search.best_params_

# Map values to the planned tunables by kind/range.
best_params["lgbm_num_leaves"] = int(sp.get("data_op__1", sp.get("lgbm_num_leaves", 31)))
best_params["lgbm_learning_rate"] = float(sp.get("data_op__0", sp.get("lgbm_learning_rate", 0.03)))

print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
