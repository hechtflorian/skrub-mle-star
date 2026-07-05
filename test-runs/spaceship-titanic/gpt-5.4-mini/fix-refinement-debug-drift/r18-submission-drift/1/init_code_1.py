
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

def add_features(df):
    out = df.copy()

    if "Name" in out.columns:
        # Correct Alone feature: passenger is alone when no last name is shared within the cabin group
        out["Alone"] = out["Cabin"].notna() & out["Name"].isna()
    else:
        out["Alone"] = False

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype("string").str.split("/", expand=True)
        if cabin_parts.shape[1] >= 3:
            out["Deck"] = cabin_parts[0]
            out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
            out["Side"] = cabin_parts[2]
        else:
            out["Deck"] = pd.NA
            out["CabinNum"] = np.nan
            out["Side"] = pd.NA

    if "Name" in out.columns:
        name_parts = out["Name"].astype("string").str.split(" ", n=1, expand=True)
        out["FirstName"] = name_parts[0]
        out["LastName"] = name_parts[1] if name_parts.shape[1] > 1 else pd.NA

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    existing_spend_cols = [c for c in spend_cols if c in out.columns]
    if existing_spend_cols:
        out["TotalSpend"] = out[existing_spend_cols].sum(axis=1)
        out["NoSpend"] = (out[existing_spend_cols].fillna(0).sum(axis=1) == 0)

    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="Accuracy",
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=random_state,
    verbose=0,
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
