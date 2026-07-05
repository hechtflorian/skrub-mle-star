
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

random_state = 42
target_col = "Transported"

train_df = pd.read_csv(os.path.join("./input", "train.csv"))

def add_features(df):
    out = df.copy()

    # Basic cabin parsing
    cabin_parts = out["Cabin"].astype("string").str.split("/", expand=True)
    if cabin_parts.shape[1] >= 3:
        out["CabinDeck"] = cabin_parts[0]
        out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
        out["CabinSide"] = cabin_parts[2]

    # Name parsing
    if "Name" in out.columns:
        out["Surname"] = out["Name"].astype("string").str.split(" ", n=1).str[-1]
        out["NameLen"] = out["Name"].astype("string").str.len()

    # Family size / group features
    if "PassengerId" in out.columns:
        out["Group"] = out["PassengerId"].astype("string").str.split("_").str[0]

    spend_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in out.columns]
    if spend_cols:
        out["TotalSpend"] = out[spend_cols].sum(axis=1)
        out["HasSpent"] = (out["TotalSpend"] > 0).astype(int)

        if "Surname" in out.columns:
            surname_group = out["Surname"].fillna("missing")
            out["FamilySpend"] = out.groupby(surname_group)[spend_cols].transform("sum").sum(axis=1)

    if "Age" in out.columns:
        out["AgeGroup"] = pd.cut(out["Age"], bins=[-np.inf, 12, 18, 30, 50, np.inf], labels=False)

    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_features)

X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

model = RandomForestClassifier(
    n_estimators=300,
    random_state=random_state,
    n_jobs=-1,
)

pred = X_train.skb.apply(vectorizer).skb.apply(
    make_pipeline(
        SimpleImputer(strategy="median"),
        model,
    ),
    y=y_train,
)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred)

if valid_pred.dtype != bool:
    valid_pred = valid_pred.astype(str)
    valid_pred = np.array([x in ("True", "1", "true", "yes") for x in valid_pred])

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
