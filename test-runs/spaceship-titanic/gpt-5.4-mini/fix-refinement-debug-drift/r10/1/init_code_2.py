
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def add_features(df):
    out = df.copy()

    cabin = out["Cabin"].fillna("U/U/U").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]

    for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["TotalSpend"] = out[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["Spent"] = (out["TotalSpend"] > 0).astype(float)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["AgeBucket"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 80, 200],
        labels=False,
        include_lowest=True,
    )
    out["CryoSleep"] = out["CryoSleep"].astype("object")
    out["VIP"] = out["VIP"].astype("object")
    out["HasCabin"] = out["Cabin"].notna().astype(float)

    return out.drop(columns=["Cabin", "Name"], errors="ignore")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state, stratify=train_df[target_col].astype(int)
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
valid_pred = np.where(valid_pred >= 0.5, True, False)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_features)
X_full = data_full_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})
test_pred = np.asarray(test_pred).ravel()
test_pred = np.where(test_pred >= 0.5, True, False)

submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": test_pred.astype(bool)})
submission.to_csv("submission.csv", index=False)
