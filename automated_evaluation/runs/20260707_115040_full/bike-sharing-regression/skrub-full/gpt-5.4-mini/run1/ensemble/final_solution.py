
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

random_state = 42
target_col = "count"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# Holdout block: bind only train_part so validation is honest
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
model = skrub.choose_from(
    {
        "ridge": __import__("sklearn.linear_model", fromlist=["Ridge"]).Ridge(alpha=1.0),
        "rf": __import__("sklearn.ensemble", fromlist=["RandomForestRegressor"]).RandomForestRegressor(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
    },
    name="model_choice",
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
learner = pred.skb.make_learner(fitted=True)
valid_pred = np.asarray(learner.predict({"data": valid_part}), dtype=float)
valid_pred = np.clip(valid_pred, 0, None)
final_validation_score = mean_squared_log_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Submission block: refit on full train_df, but drop leakage columns from features
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=[target_col, "casual", "registered"], errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred = np.asarray(full_learner.predict({"data": test_df}), dtype=float)
test_pred = np.clip(test_pred, 0, None)

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"datetime": test_df["datetime"], target_col: test_pred})
submission.to_csv("./final/submission.csv", index=False)
