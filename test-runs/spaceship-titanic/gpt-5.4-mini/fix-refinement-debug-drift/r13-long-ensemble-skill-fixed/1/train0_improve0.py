
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

    # Cabin structure
    cabin = out["Cabin"].fillna("Unknown/0/U").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0].replace("", "Unknown")
    out["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2].replace("", "Unknown")
    out["CabinKnown"] = out["Cabin"].notna().astype(int)
    out["DeckKnown"] = (out["Deck"] != "Unknown").astype(int)
    out["CabinNumKnown"] = out["Num"].notna().astype(int)

    # Spending features
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out[spend_cols] = out[spend_cols].fillna(0)
    out["TotalSpending"] = out[spend_cols].sum(axis=1)
    out["HasSpent"] = (out["TotalSpending"] > 0).astype(int)
    out["AnySpendMissing"] = out[spend_cols].isna().any(axis=1).astype(int)
    out["SpendNonZeroCount"] = (out[spend_cols] > 0).sum(axis=1)
    out["SpendMean"] = out[spend_cols].mean(axis=1)
    out["SpendMax"] = out[spend_cols].max(axis=1)
    out["SpendStd"] = out[spend_cols].std(axis=1).fillna(0.0)
    out["SpendShareRoomService"] = (
        out["RoomService"] / out["TotalSpending"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["SpendShareFoodCourt"] = (
        out["FoodCourt"] / out["TotalSpending"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Name / family features
    name = out["Name"].fillna("").astype(str)
    out["NameLen"] = name.str.len()
    out["Surname"] = name.str.split().str[-1].replace("", "Unknown")
    out["SurnameLen"] = out["Surname"].astype(str).str.len()
    out["HasSurname"] = (out["Surname"] != "Unknown").astype(int)

    # Passenger group structure
    passenger_group = out["PassengerId"].astype(str).str.split("_").str[0]
    out["Group"] = passenger_group
    group_size = passenger_group.map(passenger_group.value_counts())
    out["GroupSize"] = group_size
    out["Solo"] = (out["GroupSize"] == 1).astype(int)

    # Cabin/group consistency signals
    out["FamilySizeLike"] = out["GroupSize"].fillna(1).astype(float)
    out["GroupCabinKnownFrac"] = out.groupby(passenger_group)["CabinKnown"].transform("mean")
    out["GroupSpentAnyFrac"] = out.groupby(passenger_group)["HasSpent"].transform("mean")
    out["GroupMedianAge"] = out.groupby(passenger_group)["Age"].transform("median")
    out["AgeVsGroupMedian"] = (
        out["Age"] - out["GroupMedianAge"]
    ).replace([np.inf, -np.inf], np.nan)

    # Age features
    out["AgeKnown"] = out["Age"].notna().astype(int)
    out["AgeBin"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 65, 120],
        labels=["child", "teen", "young", "adult", "mid", "senior", "elder"],
    ).astype("object")
    out["AgeIsChild"] = (out["Age"] <= 12).astype(int)
    out["AgeIsSenior"] = (out["Age"] >= 65).astype(int)
    out["AgeSquared"] = out["Age"] ** 2

    # Per-person / ratio features
    out["SpendingPerPerson"] = (
        out["TotalSpending"] / out["GroupSize"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["AgePerGroupMember"] = (
        out["Age"] / out["GroupSize"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["LogTotalSpending"] = np.log1p(out["TotalSpending"])
    out["LogSpendingPerPerson"] = np.log1p(out["SpendingPerPerson"])

    # Missingness indicators
    out["NameKnown"] = out["Name"].notna().astype(int)
    out["AgeMissing"] = out["Age"].isna().astype(int)
    out["CabinMissing"] = out["Cabin"].isna().astype(int)
    out["GroupSizeKnown"] = out["GroupSize"].notna().astype(int)

    return out



train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col].astype(int),
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# Model 1: CatBoost baseline
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

# Model 2: LightGBM reference-style leg
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

def pred_to_proba(learner, df):
    pred = learner.predict({"data": df})
    pred = np.asarray(pred)
    if pred.ndim == 2 and pred.shape[1] > 1:
        return pred[:, 1].astype(float).ravel()
    return pred.astype(float).ravel()

valid_pred_1 = pred_to_proba(learner_1, valid_part)
valid_pred_2 = pred_to_proba(learner_2, valid_part)

valid_blend = 0.45 * valid_pred_1 + 0.55 * valid_pred_2
valid_pred = (valid_blend >= 0.5).astype(int)
y_valid = valid_part[target_col].astype(int).to_numpy()

final_validation_score = accuracy_score(y_valid, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage: refit both legs on full train and blend predictions on test
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
test_blend = 0.45 * test_pred_1 + 0.55 * test_pred_2
test_pred = (test_blend >= 0.5).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
