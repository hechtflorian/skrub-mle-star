import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Personality"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Holdout split for honest validation
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Minimal fix: remove cat_features so CatBoost does not try to match
# original categorical column names against transformed vectorized features.

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_missing_indicators(df):
    out = df.copy()
    null_cols = out.columns[out.isna().any()]
    for col in null_cols:
        out[f"{col}__is_null"] = out[col].isna().astype("int8")
    return out


data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_missing_indicators)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

cleaner = skrub.Cleaner(drop_if_constant=True)
vectorizer = skrub.TableVectorizer()

model = CatBoostClassifier(
    loss_function="Logloss",
    verbose=0,
    random_seed=random_state,
)

pred = (
    X_train.skb.apply(cleaner)
    .skb.apply(vectorizer)
    .skb.apply(model, y=y_train)
)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Block 2 — full-train refit and test export
os.makedirs("./final", exist_ok=True)

data_full = skrub.var("data", train_df)
data_full = data_full.skb.apply_func(add_missing_indicators)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = (
    X_full.skb.apply(cleaner)
    .skb.apply(vectorizer)
    .skb.apply(model, y=y_full)
)

full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({"id": test_df["id"], target_col: test_pred})
submission.to_csv("./final/submission.csv", index=False)