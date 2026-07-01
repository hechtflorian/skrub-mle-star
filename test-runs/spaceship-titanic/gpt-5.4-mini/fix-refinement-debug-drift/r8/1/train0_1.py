
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
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

    family_id = out["PassengerId"].astype(str).str.split("_").str[0]
    out["FamilyId"] = family_id
    out["SpendPerAge"] = out["TotalSpend"] / (out["Age"].fillna(out["Age"].median()) + 1.0)
    out["CabinNumBucket"] = pd.cut(
        out["Num"].fillna(-1),
        bins=[-2, 0, 50, 100, 500, 10000],
        labels=["Unknown", "Low", "Mid", "High", "VeryHigh"],
    ).astype(str)
    return out


metric_fn = accuracy_score


def build_lightgbm_graph(data_train):
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
    return X_train.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(model, y=y_train)


def build_catboost_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.03,
        loss_function="Logloss",
        verbose=0,
        random_seed=42,
    )
    return X_train.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(model, y=y_train)


data_train = skrub.var("data", train_part)
pred_lgbm = build_lightgbm_graph(data_train)
learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
valid_pred_lgbm = np.asarray(learner_lgbm.predict({"data": valid_part})).astype(float)

data_train_cb = skrub.var("data", train_part)
pred_cb = build_catboost_graph(data_train_cb)
learner_cb = pred_cb.skb.make_learner(fitted=True)
valid_pred_cb = np.asarray(learner_cb.predict({"data": valid_part})).astype(float)

blend_valid = 0.6 * valid_pred_lgbm + 0.4 * valid_pred_cb
final_validation_score = metric_fn(
    valid_part[target_col].astype(int), (blend_valid >= 0.5).astype(int)
)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

vectorizer_full_lgbm = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.StringEncoder(),
)
model_full_lgbm = LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
full_graph_lgbm = X_full.skb.apply_func(fe_func).skb.apply(vectorizer_full_lgbm).skb.apply(
    model_full_lgbm, y=y_full
)
full_learner_lgbm = full_graph_lgbm.skb.make_learner(fitted=True)
test_pred_lgbm = np.asarray(full_learner_lgbm.predict({"data": test_df})).astype(float)

vectorizer_full_cb = skrub.TableVectorizer()
model_full_cb = CatBoostClassifier(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    verbose=0,
    random_seed=42,
)
full_graph_cb = X_full.skb.apply_func(fe_func).skb.apply(vectorizer_full_cb).skb.apply(
    model_full_cb, y=y_full
)
full_learner_cb = full_graph_cb.skb.make_learner(fitted=True)
test_pred_cb = np.asarray(full_learner_cb.predict({"data": test_df})).astype(float)

blend_test = 0.6 * test_pred_lgbm + 0.4 * test_pred_cb

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": (blend_test >= 0.5).astype(bool),
    }
)
submission.to_csv("submission.csv", index=False)
