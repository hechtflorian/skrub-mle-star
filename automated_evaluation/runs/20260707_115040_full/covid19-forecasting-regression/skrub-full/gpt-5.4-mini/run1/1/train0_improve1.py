
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor

random_state = 42
test_size = 0.2
target_cols = ["ConfirmedCases", "Fatalities"]

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

# Basic preprocessing
for col in ["Province_State", "Country_Region"]:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna("")

train_df["Date"] = pd.to_datetime(train_df["Date"], errors="coerce")
train_df["Date_ordinal"] = train_df["Date"].map(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
train_df = train_df.drop(columns=["Date"], errors="ignore")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_pred = np.maximum(y_pred, 0)
    return mean_squared_log_error(y_true, y_pred) ** 0.5

scores = []

for target_col in target_cols:
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        verbosity=-1,
    )

    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    val_learner = pred.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict({"data": valid_part})
    score = rmsle(valid_part[target_col], valid_pred)
    scores.append(score)

final_validation_score = float(np.mean(scores))
print(f"Final Validation Performance: {final_validation_score}")
