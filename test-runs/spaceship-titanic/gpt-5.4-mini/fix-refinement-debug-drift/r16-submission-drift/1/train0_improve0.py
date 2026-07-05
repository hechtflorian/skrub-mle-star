
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
test_size = 0.2

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_col = "Transported"

# Ensure target is boolean/integer-friendly for CatBoost
train_df[target_col] = train_df[target_col].astype(int)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def preprocess(df):
    out = df.copy()
    for col in out.columns:
        if col == target_col:
            continue
        if col == "Cabin":
            cabin_parts = out[col].fillna("").astype(str).str.split("/", expand=True)
            out["CabinDeck"] = cabin_parts[0]
            out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
            out["CabinSide"] = cabin_parts[2]
            out = out.drop(columns=[col])
        elif col == "Name":
            out["NameLength"] = out[col].fillna("").astype(str).str.len()
            out = out.drop(columns=[col])
    return out

data_train = skrub.var("data", preprocess(train_part))
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

model = CatBoostClassifier(
    loss_function="Logloss",
    verbose=0,
    random_seed=random_state,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": preprocess(valid_part)})
valid_pred = np.asarray(valid_pred).astype(int)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
