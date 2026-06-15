
import sys
import subprocess
import warnings

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "lightgbm", "-q"])

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


import numpy as np
import skrub
import skrub.selectors as s
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

@skrub.deferred
def add_spaceship_features(df):
    out = df.copy()

    # PassengerId-derived group size
    if "PassengerId" in out.columns:
        passenger_group = (
            out["PassengerId"]
            .astype(str)
            .str.split("_", expand=False)
            .str[0]
        )
        out["passenger_group"] = passenger_group
        group_sizes = passenger_group.map(passenger_group.value_counts()).astype(float)
        out["passenger_group_size"] = group_sizes.fillna(1.0)

    # Cabin-derived deck/side features
    if "Cabin" in out.columns:
        cabin = out["Cabin"].astype(str)
        cabin_is_na = out["Cabin"].isna() | cabin.str.lower().isin(["nan", "none", ""])
        cabin_split = cabin.str.split("/", expand=True)
        out["cabin_deck"] = cabin_split[0].where(~cabin_is_na, np.nan)
        out["cabin_num"] = pd.to_numeric(cabin_split[1], errors="coerce")
        out["cabin_side"] = cabin_split[2].where(~cabin_is_na, np.nan)

    # Spend aggregates and zero-spend indicators
    spend_cols = [c for c in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"] if c in out.columns]
    if spend_cols:
        spend_frame = out[spend_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        out["total_spend"] = spend_frame.sum(axis=1)
        out["mean_spend"] = spend_frame.mean(axis=1)
        out["max_spend"] = spend_frame.max(axis=1)
        out["zero_spend_count"] = (spend_frame == 0).sum(axis=1)
        out["has_any_spend"] = (out["total_spend"] > 0).astype(float)
        out["spend_std"] = spend_frame.std(axis=1).fillna(0.0)

    # Basic family/group signal if present
    if "Name" in out.columns:
        out["name_len"] = out["Name"].astype(str).str.len().fillna(0).astype(float)

    # Clean obvious infinities and preserve graph-compatible dtypes
    out = out.replace([np.inf, -np.inf], np.nan)
    return out

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_spaceship_features)

X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Drop raw leakage/redundant fields after extracting structured features
X_train = X_train.skb.apply(
    skrub.DropCols(cols=s.cols("PassengerId", "Cabin"))
)

# Keep a single post-FE vectorizer on the remaining table
vectorizer = skrub.TableVectorizer()

X_vec = X_train.skb.apply(vectorizer)

cat_model = CatBoostClassifier(
    verbose=0,
    loss_function="Logloss",
    random_seed=42,
    depth=6,
    learning_rate=0.05,
    iterations=500,
)

lgb_model = LGBMClassifier(
    random_state=42,
    n_estimators=600,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.85,
    verbose=-1,
)

cat_pred = X_vec.skb.apply(cat_model, y=y_train)
lgb_pred = X_vec.skb.apply(lgb_model, y=y_train)

cat_learner = cat_pred.skb.make_learner(fitted=True)
lgb_learner = lgb_pred.skb.make_learner(fitted=True)

cat_valid_pred = np.asarray(cat_learner.predict({"data": valid_part})).ravel()
lgb_valid_pred = np.asarray(lgb_learner.predict({"data": valid_part})).ravel()

# Soft ensemble on probabilities/scores instead of hard voting
if cat_valid_pred.dtype == bool:
    cat_valid_pred = cat_valid_pred.astype(float)
else:
    cat_valid_pred = np.clip(cat_valid_pred.astype(float), 0.0, 1.0)

if lgb_valid_pred.dtype == bool:
    lgb_valid_pred = lgb_valid_pred.astype(float)
else:
    lgb_valid_pred = np.clip(lgb_valid_pred.astype(float), 0.0, 1.0)

ensemble_valid_pred = 0.6 * cat_valid_pred + 0.4 * lgb_valid_pred
ensemble_valid_pred = ensemble_valid_pred >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], ensemble_valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

