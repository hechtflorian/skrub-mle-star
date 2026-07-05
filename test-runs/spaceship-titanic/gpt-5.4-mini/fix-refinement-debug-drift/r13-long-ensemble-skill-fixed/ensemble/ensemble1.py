
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")


def add_features_base(df):
    out = df.copy()
    cabin = out["Cabin"].fillna("Unknown/0/U").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out[spend_cols] = out[spend_cols].fillna(0)
    out["TotalSpending"] = out[spend_cols].sum(axis=1)
    out["NameLen"] = out["Name"].fillna("").astype(str).str.len()
    out["Group"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["GroupSize"] = out["Group"].map(out["Group"].value_counts())
    out["IsAlone"] = (out["GroupSize"] == 1).astype(int)
    out["SpendingPerPerson"] = (
        out["TotalSpending"] / out["GroupSize"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_features_ref(df):
    out = df.copy()

    cabin = out["Cabin"].fillna("Unknown/0/U").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out[spend_cols] = out[spend_cols].fillna(0)
    out["TotalSpending"] = out[spend_cols].sum(axis=1)
    out["HasSpent"] = (out["TotalSpending"] > 0).astype(int)

    out["NameLen"] = out["Name"].fillna("").astype(str).str.len()
    out["Surname"] = out["Name"].fillna("").astype(str).str.split().str[-1]

    passenger_group = out["PassengerId"].astype(str).str.split("_").str[0]
    out["Group"] = passenger_group
    out["GroupSize"] = passenger_group.map(passenger_group.value_counts())
    out["Solo"] = (out["GroupSize"] == 1).astype(int)

    out["AgeBin"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 65, 120],
        labels=["child", "teen", "young", "adult", "mid", "senior", "elder"],
    ).astype("object")

    out["SpendingPerPerson"] = (
        out["TotalSpending"] / out["GroupSize"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["CabinKnown"] = out["Cabin"].notna().astype(int)
    out["NameKnown"] = out["Name"].notna().astype(int)

    return out


def pred_to_proba(learner, df):
    try:
        pred = learner.predict_proba({"data": df})
        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] > 1:
            return pred[:, 1].astype(float).ravel()
        return pred.astype(float).ravel()
    except Exception:
        pred = learner.predict({"data": df})
        pred = np.asarray(pred)
        if pred.ndim == 2 and pred.shape[1] > 1:
            return pred[:, 1].astype(float).ravel()
        return pred.astype(float).ravel()


def make_meta_features(p1, p2):
    p1 = np.asarray(p1, dtype=float).ravel()
    p2 = np.asarray(p2, dtype=float).ravel()
    abs_diff = np.abs(p1 - p2)
    mean_p = 0.5 * (p1 + p2)
    agree_flag = (abs_diff < 0.08).astype(float)
    return np.column_stack([p1, p2, abs_diff, mean_p, agree_flag])


train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col].astype(int),
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train_1 = skrub.var("data", train_part)
data_train_fe_1 = data_train_1.skb.apply_func(add_features_base)
X_train_1 = data_train_fe_1.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train_1 = data_train_fe_1[target_col].skb.mark_as_y()

cat_cols_1 = ["HomePlanet", "CryoSleep", "VIP", "Deck", "Side", "Destination", "Group"]
for col in cat_cols_1:
    X_train_1 = X_train_1.assign(**{col: X_train_1[col].astype("category")})

vectorizer_1 = skrub.TableVectorizer()
model_1 = CatBoostClassifier(
    iterations=1200,
    learning_rate=0.03,
    depth=8,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
    allow_writing_files=False,
)

pred_graph_1 = X_train_1.skb.apply(vectorizer_1).skb.apply(model_1, y=y_train_1)
learner_1 = pred_graph_1.skb.make_learner(fitted=True)

data_train_2 = skrub.var("data", train_part)
data_train_fe_2 = data_train_2.skb.apply_func(add_features_ref)
X_train_2 = data_train_fe_2.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train_2 = data_train_fe_2[target_col].skb.mark_as_y()

cat_cols_2 = [
    "HomePlanet",
    "CryoSleep",
    "VIP",
    "Deck",
    "Side",
    "Destination",
    "Group",
    "AgeBin",
    "Surname",
]
for col in cat_cols_2:
    X_train_2 = X_train_2.assign(**{col: X_train_2[col].astype("category")})

vectorizer_2 = skrub.TableVectorizer()
model_2 = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred_graph_2 = X_train_2.skb.apply(vectorizer_2).skb.apply(model_2, y=y_train_2)
learner_2 = pred_graph_2.skb.make_learner(fitted=True)

valid_pred_1 = pred_to_proba(learner_1, valid_part)
valid_pred_2 = pred_to_proba(learner_2, valid_part)

meta_X_valid = make_meta_features(valid_pred_1, valid_pred_2)
y_valid = valid_part[target_col].astype(int).to_numpy()

meta_plain = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state)
meta_plain.fit(np.column_stack([valid_pred_1, valid_pred_2]), y_valid)
plain_valid_pred = (
    meta_plain.predict_proba(np.column_stack([valid_pred_1, valid_pred_2]))[:, 1] >= 0.5
).astype(int)
plain_acc = accuracy_score(y_valid, plain_valid_pred)

meta_agree = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state)
meta_agree.fit(meta_X_valid, y_valid)
agree_raw = meta_agree.predict_proba(meta_X_valid)[:, 1]
valid_avg = 0.5 * (valid_pred_1 + valid_pred_2)
valid_blend = np.where(np.abs(valid_pred_1 - valid_pred_2) < 0.08, valid_avg, agree_raw)
agree_valid_pred = (valid_blend >= 0.5).astype(int)
agree_acc = accuracy_score(y_valid, agree_valid_pred)

if agree_acc >= plain_acc:
    final_meta_mode = "agree_fallback"
    meta_model = meta_agree
    valid_pred = agree_valid_pred
else:
    final_meta_mode = "plain_stack"
    meta_model = meta_plain
    valid_pred = plain_valid_pred

final_validation_score = accuracy_score(y_valid, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

data_full_1 = skrub.var("data", train_df)
data_full_fe_1 = data_full_1.skb.apply_func(add_features_base)
X_full_1 = data_full_fe_1.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_full_1 = data_full_fe_1[target_col].skb.mark_as_y()
for col in cat_cols_1:
    X_full_1 = X_full_1.assign(**{col: X_full_1[col].astype("category")})
full_pred_graph_1 = X_full_1.skb.apply(vectorizer_1).skb.apply(model_1, y=y_full_1)
full_learner_1 = full_pred_graph_1.skb.make_learner(fitted=True)

data_full_2 = skrub.var("data", train_df)
data_full_fe_2 = data_full_2.skb.apply_func(add_features_ref)
X_full_2 = data_full_fe_2.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_full_2 = data_full_fe_2[target_col].skb.mark_as_y()
for col in cat_cols_2:
    X_full_2 = X_full_2.assign(**{col: X_full_2[col].astype("category")})
full_pred_graph_2 = X_full_2.skb.apply(vectorizer_2).skb.apply(model_2, y=y_full_2)
full_learner_2 = full_pred_graph_2.skb.make_learner(fitted=True)

test_pred_1 = pred_to_proba(full_learner_1, test_df)
test_pred_2 = pred_to_proba(full_learner_2, test_df)

if final_meta_mode == "plain_stack":
    test_meta_X = np.column_stack([test_pred_1, test_pred_2])
    test_meta_prob = meta_model.predict_proba(test_meta_X)[:, 1]
    test_pred = (test_meta_prob >= 0.5).astype(bool)
else:
    test_meta_X = make_meta_features(test_pred_1, test_pred_2)
    test_meta_prob = meta_model.predict_proba(test_meta_X)[:, 1]
    test_avg = 0.5 * (test_pred_1 + test_pred_2)
    test_final_prob = np.where(np.abs(test_pred_1 - test_pred_2) < 0.08, test_avg, test_meta_prob)
    test_pred = (test_final_prob >= 0.5).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
