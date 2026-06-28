
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

target_col = "Transported"
train_df = pd.read_csv("./input/train.csv")

def add_structured_features(df):
    out = df.copy()

    if "Cabin" in out.columns:
        cabin = out["Cabin"].astype("string")
        cabin_parts = cabin.str.split("/", expand=True)
        if cabin_parts.shape[1] >= 1:
            out["Deck"] = cabin_parts[0]
        if cabin_parts.shape[1] >= 2:
            out["Num"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        if cabin_parts.shape[1] >= 3:
            out["Side"] = cabin_parts[2]

    family_cols = ["PassengerId", "Name"]
    if "PassengerId" in out.columns:
        grp = out["PassengerId"].astype("string").str.split("_").str[0]
        family_size = out.groupby(grp, dropna=False)["PassengerId"].transform("size")
        out["FamilySize"] = family_size
        out["Solo"] = (family_size == 1)

    spend_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in out.columns]
    if spend_cols:
        spend_df = out[spend_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["TotalSpend"] = spend_df.sum(axis=1)
        out["SpentAny"] = (out["TotalSpend"] > 0).astype(bool)

    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_structured_features)

X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
predictor = X_train.skb.apply(vectorizer).skb.apply(
    HistGradientBoostingClassifier(random_state=42), y=y_train
)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
