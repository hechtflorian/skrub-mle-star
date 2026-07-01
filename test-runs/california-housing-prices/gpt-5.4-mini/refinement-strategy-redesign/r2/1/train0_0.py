
import os
import pandas as pd
import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

DATA_DIR = "./input"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

target_col = "median_house_value"

# Build DataOps graph
data = skrub.var("data", train_df)

# Use errors="ignore" to avoid KeyError if the target is absent in any intermediate frame
X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

model = HistGradientBoostingRegressor(random_state=0)
pred_graph = X_vec.skb.apply(model, y=y)

# Compile fitted learner
learner = pred_graph.skb.make_learner(fitted=True)

# Validation on train split using a simple holdout
rng = np.random.RandomState(0)
indices = np.arange(len(train_df))
rng.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx, val_idx = indices[:split], indices[split:]

train_subset = train_df.iloc[train_idx].copy()
val_subset = train_df.iloc[val_idx].copy()

# Fit on training subset using the same DataOps structure
train_data = skrub.var("data", train_subset)
train_X = train_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
train_y = train_data[target_col].skb.mark_as_y()
train_X_vec = train_X.skb.apply(vectorizer)
train_pred_graph = train_X_vec.skb.apply(model, y=train_y)
trained_learner = train_pred_graph.skb.make_learner(fitted=True)

# Predict on validation features only; do NOT drop median_house_value if it is already absent
val_features = val_subset.drop(columns=[target_col], errors="ignore")
val_pred = trained_learner.predict({"data": val_features})

final_validation_score = mean_squared_error(val_subset[target_col], val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Train on full data and predict test
full_learner = pred_graph.skb.make_learner(fitted=True)
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
