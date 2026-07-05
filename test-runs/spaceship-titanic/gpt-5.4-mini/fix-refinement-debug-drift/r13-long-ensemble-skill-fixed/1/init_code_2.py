
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

def add_features(df):
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

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col].astype(int),
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)

X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

cat_cols = ["HomePlanet", "CryoSleep", "VIP", "Deck", "Side", "Destination", "Group", "AgeBin", "Surname"]
for col in cat_cols:
    X_train = X_train.assign(**{col: X_train[col].astype("category")})

vectorizer = skrub.TableVectorizer()
model = LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred_graph.skb.make_learner(fitted=True)

valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).astype(int).ravel()
y_valid = valid_part[target_col].astype(int).to_numpy()

final_validation_score = accuracy_score(y_valid, valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_features)

X_full = data_full_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].skb.mark_as_y()

for col in cat_cols:
    X_full = X_full.assign(**{col: X_full[col].astype("category")})

full_pred_graph = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred_graph.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).astype(int).ravel().astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
