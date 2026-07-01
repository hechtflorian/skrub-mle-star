
import json
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df[target_col],
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
    out["HasSpent"] = (out["TotalSpend"] > 0).astype(int)

    out["AgeGroup"] = pd.cut(
        out["Age"].fillna(-1),
        bins=[-2, 0, 12, 18, 30, 45, 60, 200],
        labels=["Unknown", "Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior"],
    ).astype(str)

    out["FamilySizeFlag"] = (
        out["PassengerId"].astype(str).str.split("_").str[1].astype(int) > 0
    ).astype(int)
    out["LogTotalSpend"] = np.log1p(out["TotalSpend"])
    return out

metric_fn = accuracy_score

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.StringEncoder(),
)

model = LGBMClassifier(
    n_estimators=skrub.choose_int(800, 1600, n_steps=5, default=1200, name="n_estimators"),
    learning_rate=skrub.choose_float(0.01, 0.04, log=True, default=0.02, name="learning_rate"),
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

pred_graph = X_train.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(model, y=y_train)
search = pred_graph.skb.make_randomized_search(n_iter=4, n_jobs=1, random_state=42, fitted=True)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
final_validation_score = metric_fn(valid_part[target_col].astype(int), np.asarray(valid_pred).astype(int))
print(f"Final Validation Performance: {final_validation_score}")

best_params = {
    "n_estimators": int(search.best_params_.get("data_op__0", 1200)),
    "learning_rate": float(search.best_params_.get("data_op__1", 0.02)),
}
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
