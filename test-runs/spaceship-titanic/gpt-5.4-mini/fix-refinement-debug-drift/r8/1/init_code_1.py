
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def fe_func(df):
    out = df.copy()
    out["Cabin"] = out["Cabin"].fillna("X/0/X").astype(str)
    cabin_split = out["Cabin"].str.split("/", expand=True)
    out["Deck"] = cabin_split[0]
    out["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    out["Side"] = cabin_split[2]
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["TotalSpend"] = out[spend_cols].fillna(0).sum(axis=1)
    out["IsAlone"] = (out["TotalSpend"] == 0).astype(int)
    out["AgeGroup"] = pd.cut(
        out["Age"].fillna(-1),
        bins=[-2, 0, 12, 18, 30, 45, 60, 200],
        labels=["Unknown", "Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior"],
    ).astype(str)
    out["HasSpent"] = (out["TotalSpend"] > 0).astype(int)
    return out

def build_graph(data_train):
    data_train = data_train.skb.apply_func(fe_func)
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    encoder = skrub.TableVectorizer()
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        verbose=0,
        random_seed=42,
    )
    return X.skb.apply(encoder).skb.apply(model, y=y)

data_train = skrub.var("data", train_part)
pred = build_graph(data_train)
learner = pred.skb.make_learner(fitted=True)
valid_pred = learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col].astype(int), np.asarray(valid_pred).astype(int))
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.skb.apply_func(fe_func).drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full.skb.apply_func(fe_func)[target_col].skb.mark_as_y()
encoder_full = skrub.TableVectorizer()
model_full = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    verbose=0,
    random_seed=42,
)
full_pred = X_full.skb.apply(encoder_full).skb.apply(model_full, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": np.asarray(test_pred).astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
