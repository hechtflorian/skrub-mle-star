
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
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

    out["AgeGroup"] = pd.cut(
        out["Age"].fillna(-1),
        bins=[-2, 0, 12, 18, 30, 45, 60, 200],
        labels=["Unknown", "Child", "Teen", "YoungAdult", "Adult", "MiddleAge", "Senior"],
    ).astype(str)

    out["FamilySizeFlag"] = (
        out["PassengerId"].astype(str).str.split("_").str[1].astype(int) > 0
    ).astype(int)
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
    n_estimators=1200,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

pred_graph = X_train.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(model, y=y_train)
learner = pred_graph.skb.make_learner(fitted=True)
valid_pred = learner.predict({"data": valid_part})
final_validation_score = metric_fn(valid_part[target_col].astype(int), np.asarray(valid_pred).astype(int))
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

vectorizer_full = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.StringEncoder(),
)
model_full = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)

full_graph = X_full.skb.apply_func(fe_func).skb.apply(vectorizer_full).skb.apply(model_full, y=y_full)
full_learner = full_graph.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": np.asarray(test_pred).astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
