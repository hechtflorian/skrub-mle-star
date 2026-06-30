
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "Transported"

# Preprocess with the smallest possible fix for SpentTotal

import numpy as np
import pandas as pd
import skrub
import skrub.selectors as s
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def preprocess(df):
    out = df.copy()

    numeric_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    present_numeric = [c for c in numeric_cols if c in out.columns]
    if present_numeric:
        out[present_numeric] = out[present_numeric].apply(pd.to_numeric, errors="coerce")
        out["SpentTotal"] = out[present_numeric].fillna(0).sum(axis=1)
    else:
        out["SpentTotal"] = 0.0

    out["HasSpent"] = (out["SpentTotal"].fillna(0) > 0).astype("int64")
    out["SpentAny"] = out["HasSpent"]
    out["SpentLog1p"] = np.log1p(out["SpentTotal"].fillna(0).clip(lower=0))

    if "Cabin" in out.columns:
        cabin_split = out["Cabin"].astype(str).str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            deck = cabin_split[0].replace("nan", np.nan)
            num = pd.to_numeric(cabin_split[1], errors="coerce")
            side = cabin_split[2].replace("nan", np.nan)

            out["CabinDeck"] = deck
            out["CabinNum"] = num
            out["CabinSide"] = side
            out["CabinDeckSide"] = deck.astype(str).fillna("Missing") + "_" + side.astype(str).fillna("Missing")
            out["CabinNumParity"] = np.where(num.notna(), (num.fillna(0).astype(int) % 2), np.nan)
            out["CabinNumBin"] = pd.cut(
                num,
                bins=[-np.inf, 100, 500, 1000, np.inf],
                labels=["small", "mid", "large", "xlarge"],
                include_lowest=True,
            )
        else:
            out["CabinDeck"] = np.nan
            out["CabinNum"] = np.nan
            out["CabinSide"] = np.nan
            out["CabinDeckSide"] = np.nan
            out["CabinNumParity"] = np.nan
            out["CabinNumBin"] = np.nan
        out = out.drop(columns=["Cabin"])

    if "PassengerId" in out.columns:
        pid = out["PassengerId"].astype(str)
        pid_split = pid.str.split("_", expand=True)
        if pid_split.shape[1] >= 2:
            out["PassengerGroup"] = pid_split[0]
            out["PassengerGroupNum"] = pd.to_numeric(pid_split[0], errors="coerce")
            out["PassengerWithinGroup"] = pd.to_numeric(pid_split[1], errors="coerce")
        else:
            out["PassengerGroup"] = np.nan
            out["PassengerGroupNum"] = np.nan
            out["PassengerWithinGroup"] = np.nan

    if "Name" in out.columns:
        out["NameLen"] = out["Name"].astype(str).str.len()
        out = out.drop(columns=["Name"])

    out = skrub.Cleaner(drop_if_constant=True).fit_transform(out)
    return out


train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_fe = data_train.skb.apply_func(preprocess)

X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()

low_card = (s.categorical() | (s.string() & s.cardinality_below(12))) - s.cols(
    "PassengerId", "Name", "Cabin", "PassengerGroup"
)
high_card = s.string() & ~s.cardinality_below(12)

X_num = X.skb.select(s.numeric())

X_low = X.skb.select(low_card).skb.apply(
    skrub.ApplyToCols(skrub.ToCategorical(), cols=s.all(), allow_reject=True)
)

X_high = X.skb.select(high_card).skb.apply(
    skrub.ApplyToCols(skrub.MinHashEncoder(n_components=8), cols=s.all(), allow_reject=True)
)

X_rem = X.skb.select(s.all() - s.numeric() - low_card - high_card)
X_rem_enc = X_rem.skb.apply(skrub.TableVectorizer())

X_all = X_num.skb.concat([X_low, X_high, X_rem_enc], axis=1)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    min_samples_split=4,
    random_state=42,
    n_jobs=-1,
)

predictor = X_all.skb.apply(model, y=y)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = metric_fn(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

train_df = preprocess(train_df)
test_df = preprocess(test_df)

# Holdout split
train_part, valid_part = train_test_split(
    train_df, test_size=0.2, random_state=42, stratify=train_df[target_col]
)

# DataOps graph
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=500,
    depth=6,
    learning_rate=0.05,
    random_seed=42,
    verbose=0
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)
learner = pred.skb.make_learner(fitted=True)

valid_pred = learner.predict({"data": valid_part})
valid_pred = np.array(valid_pred)
if valid_pred.dtype != bool and valid_pred.dtype != np.bool_:
    valid_pred = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Train on full data and create submission
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})
test_pred = np.array(test_pred)
if test_pred.dtype != bool and test_pred.dtype != np.bool_:
    test_pred = test_pred > 0.5

submission = pd.DataFrame({
    "PassengerId": pd.read_csv("./input/test.csv")["PassengerId"],
    "Transported": test_pred.astype(bool)
})
submission.to_csv("submission.csv", index=False)
