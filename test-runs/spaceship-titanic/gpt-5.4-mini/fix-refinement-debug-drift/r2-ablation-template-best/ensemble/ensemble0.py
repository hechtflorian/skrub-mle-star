
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

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
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)

cat_model = CatBoostClassifier(
    iterations=1200,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=0,
    random_seed=42,
    allow_writing_files=False,
)

lgbm_graph = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)
cat_graph = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

lgbm_learner = lgbm_graph.skb.make_learner(fitted=True)
cat_learner = cat_graph.skb.make_learner(fitted=True)

valid_pred_lgbm = np.asarray(lgbm_learner.predict({"data": valid_part})).astype(float)
valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part})).astype(float)

best_score = -1.0
best_w = 0.5
best_threshold = 0.5

for w in np.arange(0.0, 1.0001, 0.05):
    valid_blend = w * valid_pred_lgbm + (1.0 - w) * valid_pred_cat
    for threshold in [0.45, 0.5, 0.55]:
        valid_pred = (valid_blend >= threshold).astype(int)
        score = accuracy_score(valid_part[target_col].astype(int), valid_pred)
        if score > best_score:
            best_score = score
            best_w = float(w)
            best_threshold = float(threshold)

valid_blend = best_w * valid_pred_lgbm + (1.0 - best_w) * valid_pred_cat
valid_pred = (valid_blend >= best_threshold).astype(int)
final_validation_score = accuracy_score(valid_part[target_col].astype(int), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_features)

X_full = data_full_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].astype(int).skb.mark_as_y()

full_lgbm_graph = X_full.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_full)
full_cat_graph = X_full.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_full)

full_lgbm_learner = full_lgbm_graph.skb.make_learner(fitted=True)
full_cat_learner = full_cat_graph.skb.make_learner(fitted=True)

test_pred_lgbm = np.asarray(full_lgbm_learner.predict({"data": test_df})).astype(float)
test_pred_cat = np.asarray(full_cat_learner.predict({"data": test_df})).astype(float)

test_blend = best_w * test_pred_lgbm + (1.0 - best_w) * test_pred_cat
test_pred = (test_blend >= best_threshold).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
