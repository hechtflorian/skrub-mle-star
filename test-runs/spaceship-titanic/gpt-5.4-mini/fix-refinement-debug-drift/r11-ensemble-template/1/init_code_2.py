
import os
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def add_features(df):
    out = df.copy()

    cabin = out["Cabin"].fillna("Z/0/Z").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = cabin[1]
    out["Side"] = cabin[2]

    spend_cols = ["Spa", "VRDeck", "RoomService", "FoodCourt", "ShoppingMall"]
    for col in spend_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    out["TotalSpend"] = out[spend_cols].sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)

    age = pd.to_numeric(out["Age"], errors="coerce")
    out["Age"] = age
    out["IsMinor"] = (age.fillna(age.median()) < 18).astype(int)

    out["CabinNum"] = pd.to_numeric(out["Num"], errors="coerce")
    out["CabinNum"] = out["CabinNum"].replace([np.inf, -np.inf], np.nan)

    out["HasName"] = out["Name"].notna().astype(int)

    out["GroupSize"] = out["PassengerId"].astype(str).str.split("_").str[0].map(
        out["PassengerId"].astype(str).str.split("_").str[0].value_counts()
    )
    out["GroupSize"] = pd.to_numeric(out["GroupSize"], errors="coerce").fillna(1)

    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)

X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbosity=-1,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)

valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
valid_labels = (valid_pred > 0.5).astype(int)

final_validation_score = accuracy_score(
    valid_part[target_col].astype(int),
    valid_labels,
)

print(f"Final Validation Performance: {final_validation_score}")
