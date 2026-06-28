
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def prep(df):
    df = df.copy()
    cabin = df["Cabin"].astype(str).str.split("/", expand=True)
    df["CabinDeck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["CabinSide"] = cabin[2]
    df["NameLen"] = df["Name"].astype(str).str.len()
    df = df.drop(columns=["Cabin", "Name"], errors="ignore")
    df["CryoSleep"] = df["CryoSleep"].astype("float")
    df["VIP"] = df["VIP"].astype("float")
    df["HomePlanet"] = df["HomePlanet"].astype("category").cat.codes.replace(-1, np.nan)
    df["Destination"] = df["Destination"].astype("category").cat.codes.replace(-1, np.nan)
    df["CabinDeck"] = df["CabinDeck"].astype("category").cat.codes.replace(-1, np.nan)
    df["CabinSide"] = df["CabinSide"].astype("category").cat.codes.replace(-1, np.nan)
    return df

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

model = RandomForestClassifier(random_state=42, n_estimators=300, n_jobs=-1)
pred = X_train.skb.apply_func(prep).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).astype(bool)
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
