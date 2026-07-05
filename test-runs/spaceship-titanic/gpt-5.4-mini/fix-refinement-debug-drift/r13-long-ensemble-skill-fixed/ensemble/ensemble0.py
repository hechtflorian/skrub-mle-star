
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    except Exception:
        pred = learner.predict({"data": df})
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:
        return pred[:, 1].astype(float).ravel()
    return pred.astype(float).ravel()


def confidence_aware_blend(p1, p2, w1, strong_gap=0.18):
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    base = w1 * p1 + (1.0 - w1) * p2
    agree = np.abs(p1 - p2) <= strong_gap
    out = base.copy()
    out[agree] = 0.5 * (p1[agree] + p2[agree])
    out[~agree] = 0.35 * p1[~agree] + 0.65 * p2[~agree]
    return out


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
y_valid = valid_part[target_col].astype(int).to_numpy()

weight_grid = [0.35, 0.40, 0.45, 0.50]
best_weight = None
best_threshold = 0.5
best_score = -1.0
best_use_confidence = False

for w in weight_grid:
    blend_plain = w * valid_pred_1 + (1.0 - w) * valid_pred_2
    for use_confidence in [False, True]:
        blend = (
            confidence_aware_blend(valid_pred_1, valid_pred_2, w)
            if use_confidence
            else blend_plain
        )
        for thr in [0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53]:
            valid_pred = (blend >= thr).astype(int)
            score = accuracy_score(y_valid, valid_pred)
            if score > best_score:
                best_score = score
                best_weight = w
                best_threshold = thr
                best_use_confidence = use_confidence

print(f"Chosen blend weight (CatBoost): {best_weight}")
print(f"Chosen threshold: {best_threshold}")
print(f"Chosen confidence-aware blending: {best_use_confidence}")
final_validation_score = best_score
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

if best_use_confidence:
    test_blend = confidence_aware_blend(test_pred_1, test_pred_2, best_weight)
else:
    test_blend = best_weight * test_pred_1 + (1.0 - best_weight) * test_pred_2

test_pred = (test_blend >= best_threshold).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
