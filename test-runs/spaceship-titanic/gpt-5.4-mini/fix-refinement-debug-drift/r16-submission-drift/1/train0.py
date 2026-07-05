
import os
import glob
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

train_path = glob.glob(os.path.join("./input", "train.csv"))[0]
train_df = pd.read_csv(train_path)

# Keep original workflow structure; only fix categorical non-object handling in preprocess().
def preprocess(df):
    df = df.copy()

    # Create AgeGroup if not already present; keep it categorical and do not send it through median fill.
    if "AgeGroup" not in df.columns and "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 30, 50, np.inf],
            labels=["Child", "Teen", "YoungAdult", "Adult", "Senior"],
        )

    for col in df.columns:
        if col == target_col:
            continue

        # BUGFIX: Categorical columns are not numeric even if they are not object dtype.
        # Handle them separately with mode fill; only numeric columns use median.
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0] if not df[col].mode(dropna=True).empty else "Unknown")

    # Preserve existing downstream compatibility: categorical -> codes for model input
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == object:
            df[col] = df[col].astype("category").cat.codes.replace(-1, np.nan)

    return df

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps graph preserved
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Apply preprocessing inside the graph, then fit CatBoostClassifier backbone.
X_proc = X_train.skb.apply_func(preprocess)

model = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.08,
    loss_function="Logloss",
    random_seed=random_state,
    verbose=0,
)

pred = X_proc.skb.apply(model, y=y_train)

val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

# CatBoostClassifier may output probabilities; convert to labels if needed
valid_pred_arr = np.asarray(valid_pred)
if valid_pred_arr.ndim > 1:
    valid_pred_arr = valid_pred_arr[:, 0]
if valid_pred_arr.dtype != bool and valid_pred_arr.dtype != np.bool_:
    valid_pred_labels = valid_pred_arr >= 0.5
else:
    valid_pred_labels = valid_pred_arr.astype(bool)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred_labels)
print(f"Final Validation Performance: {final_validation_score}")
