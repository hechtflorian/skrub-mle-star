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
metric_label = "accuracy"

def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred_graph = build_graph(data_train)
    learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = metric_fn(valid_part[target_col].astype(int), np.asarray(valid_pred).astype(int))
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


def build_baseline(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
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
    return X.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(model, y=y)


def build_no_fe(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
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
    return X.skb.apply(vectorizer).skb.apply(model, y=y)


def build_drop_raw_cabin(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
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

    def fe_drop_cabin(df):
        out = fe_func(df)
        return out.drop(columns=["Cabin"], errors="ignore")

    return X.skb.apply_func(fe_drop_cabin).skb.apply(vectorizer).skb.apply(model, y=y)


def build_drop_spend_features(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
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

    def fe_drop_spend(df):
        out = fe_func(df)
        return out.drop(columns=["TotalSpend", "LogTotalSpend", "IsAlone", "HasSpent"], errors="ignore")

    return X.skb.apply_func(fe_drop_spend).skb.apply(vectorizer).skb.apply(model, y=y)


scores = {}
scores["baseline"] = score_variant("baseline", build_baseline)
scores["no_fe"] = score_variant("no_fe", build_no_fe)
scores["drop_raw_cabin"] = score_variant("drop_raw_cabin", build_drop_raw_cabin)
scores["drop_spend_features"] = score_variant("drop_spend_features", build_drop_spend_features)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")