
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

def feature_engineer(df):
    out = df.copy()

    if "PassengerId" in out.columns:
        pid_parts = out["PassengerId"].astype(str).str.split("_", n=1, expand=True)
        out["PassengerGroup"] = pd.to_numeric(pid_parts[0], errors="coerce")
        out["PassengerNumber"] = pd.to_numeric(pid_parts[1], errors="coerce") if pid_parts.shape[1] > 1 else np.nan

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype(str).str.split("/", n=2, expand=True)
        out["CabinDeck"] = cabin_parts[0] if cabin_parts.shape[1] > 0 else np.nan
        out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce") if cabin_parts.shape[1] > 1 else np.nan
        out["CabinSide"] = cabin_parts[2] if cabin_parts.shape[1] > 2 else np.nan

    if "Age" in out.columns:
        out["AgeGroup"] = pd.cut(
            out["Age"],
            bins=[-np.inf, 12, 18, 30, 50, np.inf],
            labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
            include_lowest=True,
        ).astype(str)

    if "Name" in out.columns:
        out["NameLen"] = out["Name"].astype(str).str.len()

    if "NameLen" in out.columns:
        out = out.drop(columns=["NameLen"], errors="ignore")

    return out

train_df = feature_engineer(train_df)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

model = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=random_state,
    n_jobs=1,
    verbose=-1,
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
if valid_pred.dtype != bool:
    valid_pred = valid_pred >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
