
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def fe_func(df):
    out = df.copy()

    if "Cabin" in out.columns:
        cabin = out["Cabin"].astype("string")
        cabin_parts = cabin.str.split("/", expand=True)
        if cabin_parts.shape[1] >= 3:
            out["CabinDeck"] = cabin_parts[0].fillna("Unknown").astype("string")
            out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
            out["CabinSide"] = cabin_parts[2].fillna("Unknown").astype("string")

    if "Name" in out.columns:
        name = out["Name"].astype("string")
        out["NameLength"] = name.fillna("").str.len().astype("float")

    if "Age" in out.columns:
        age_bins = [-np.inf, 12, 18, 30, 50, np.inf]
        age_labels = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]
        out["AgeGroup"] = pd.cut(
            pd.to_numeric(out["Age"], errors="coerce"),
            bins=age_bins,
            labels=age_labels,
            include_lowest=True,
        ).astype("string")

    numeric_cols = [
        c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in out.columns
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if numeric_cols:
        out["TotalSpend"] = out[numeric_cols].sum(axis=1, min_count=1)
        out["HasSpend"] = (out["TotalSpend"].fillna(0) > 0).astype("int64")

    return out

data_train = skrub.var("data", train_part)
data_fe = data_train.skb.apply_func(fe_func)

X_train = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
predictor = X_train.skb.apply(vectorizer).skb.apply(
    HistGradientBoostingClassifier(random_state=random_state, max_depth=6, learning_rate=0.08),
    y=y_train,
)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred)
if valid_pred.dtype != bool:
    valid_pred = valid_pred.astype(int) > 0.5

final_validation_score = accuracy_score(valid_part[target_col].astype(bool), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
