
import os
import warnings
import numpy as np
import pandas as pd
import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


def fe_func(df):
    df = df.copy()

    cabin = df["Cabin"].astype("string")
    cabin_split = cabin.str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["Side"] = cabin_split[2]

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for c in spend_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["TotalSpent"] = df[spend_cols].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpent"] == 0).astype(int)
    df["IsChild"] = (pd.to_numeric(df["Age"], errors="coerce") < 13).astype(int)

    name = df["Name"].astype("string")
    _surname = name.str.split().str[-1]
    group_id = df["PassengerId"].astype("string").str.split("_").str[0]
    df["GroupSize"] = group_id.map(group_id.value_counts())

    return df


train_df = fe_func(train_df)
test_df = fe_func(test_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

feat_cols = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    "Age",
    "VIP",
    "Deck",
    "Num",
    "Side",
    "TotalSpent",
    "NoSpend",
    "IsChild",
    "GroupSize",
]

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_1 = skrub.TableVectorizer()
model_1 = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
)
pred_chain_1 = X_train[feat_cols].skb.apply(vectorizer_1).skb.apply(model_1, y=y_train)
learner_1 = pred_chain_1.skb.make_learner(fitted=True)

vectorizer_2 = skrub.TableVectorizer()
model_2 = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbose=-1,
    ),
)
pred_chain_2 = X_train[feat_cols].skb.apply(vectorizer_2).skb.apply(model_2, y=y_train)
learner_2 = pred_chain_2.skb.make_learner(fitted=True)


def to_probabilities(learner, df):
    pred = learner.predict({"data": df})
    pred = np.asarray(pred)
    if pred.ndim > 1:
        pred = pred[:, -1]
    if pred.dtype == bool:
        return pred.astype(float)
    unique_vals = set(np.unique(pred).tolist())
    if unique_vals <= {0, 1}:
        return pred.astype(float)
    return pd.Series(pred).astype(str).isin(["True", "true", "1"]).astype(float).to_numpy()


valid_p1 = to_probabilities(learner_1, valid_part)
valid_p2 = to_probabilities(learner_2, valid_part)

valid_blend = 0.5 * valid_p1 + 0.5 * valid_p2
valid_labels = (valid_blend >= 0.5).astype(bool)

final_validation_score = accuracy_score(valid_part[target_col].values, valid_labels)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred_chain_1 = X_full[feat_cols].skb.apply(vectorizer_1).skb.apply(model_1, y=y_full)
full_learner_1 = full_pred_chain_1.skb.make_learner(fitted=True)

full_pred_chain_2 = X_full[feat_cols].skb.apply(vectorizer_2).skb.apply(model_2, y=y_full)
full_learner_2 = full_pred_chain_2.skb.make_learner(fitted=True)

test_p1 = to_probabilities(full_learner_1, test_df)
test_p2 = to_probabilities(full_learner_2, test_df)

test_blend = 0.5 * test_p1 + 0.5 * test_p2
test_labels = (test_blend >= 0.5).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_labels,
    }
)
submission.to_csv("submission.csv", index=False)
