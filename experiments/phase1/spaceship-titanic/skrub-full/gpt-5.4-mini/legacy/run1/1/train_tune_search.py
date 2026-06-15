
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

# Minimal cleaning / feature engineering
def preprocess(df):
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

train_df = preprocess(train_df)

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Keep the current DataOps preprocessing and TableVectorizer frozen.
# Tune only the model-side capacity/regularization in-graph.
model = HistGradientBoostingClassifier(
    random_state=42,
    max_depth=skrub.choose_int(3, 6, default=3, name="max_depth"),
    learning_rate=skrub.choose_float(0.03, 0.12, log=True, default=0.06, name="learning_rate"),
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)

search = pred.skb.make_randomized_search(n_iter=4, n_jobs=1, random_state=42, fitted=True)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).astype(float)
y_true = valid_part[target_col].astype(float).to_numpy()
final_validation_score = mean_squared_error(y_true, valid_pred) ** 0.5

best_params = {}
for k, v in search.best_params_.items():
    if hasattr(v, "item"):
        v = v.item()
    best_params[k] = v

print(f"Final Validation Performance: {final_validation_score}")
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
