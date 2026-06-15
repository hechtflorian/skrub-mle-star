import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load data
train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

# Baseline preprocessing from current solution
def preprocess_base(df):
    df = df.copy()
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype("string").str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
        df = df.drop(columns=["Cabin"], errors="ignore")
    if "Name" in df.columns:
        df["NameLen"] = df["Name"].astype("string").str.len()
        df = df.drop(columns=["Name"], errors="ignore")
    return df

# Ablation: disable NameLen feature only
def preprocess_no_namelen(df):
    df = df.copy()
    if "Cabin" in df.columns:
        cabin_split = df["Cabin"].astype("string").str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
        df = df.drop(columns=["Cabin"], errors="ignore")
    if "Name" in df.columns:
        df = df.drop(columns=["Name"], errors="ignore")
    return df

# Ablation: disable Cabin split features only
def preprocess_no_cabin(df):
    df = df.copy()
    if "Cabin" in df.columns:
        df = df.drop(columns=["Cabin"], errors="ignore")
    if "Name" in df.columns:
        df["NameLen"] = df["Name"].astype("string").str.len()
        df = df.drop(columns=["Name"], errors="ignore")
    return df

def run_variant(variant_name, preprocess_fn):
    train_df_p = preprocess_fn(train_df)

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df_p)), test_size=0.2, random_state=42
    )
    train_part = train_df_p.iloc[train_idx].copy()
    valid_part = train_df_p.iloc[valid_idx].copy()

    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    model = HistGradientBoostingClassifier(random_state=42)
    pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)
    val_learner = pred.skb.make_learner(fitted=True)

    valid_pred = val_learner.predict({"data": valid_part})
    valid_pred = np.asarray(valid_pred).astype(bool)
    score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score

scores = {}
scores["baseline"] = run_variant("baseline", preprocess_base)
scores["no_namelen"] = run_variant("no_namelen", preprocess_no_namelen)
scores["no_cabin"] = run_variant("no_cabin", preprocess_no_cabin)

best_variant = max(scores, key=scores.get)
print(f"Best ablation variant: {best_variant} | accuracy: {scores[best_variant]}")