

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

# Validation on train split using a simple holdout
rng = np.random.RandomState(0)
indices = np.arange(len(train_df))
rng.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx, val_idx = indices[:split], indices[split:]

train_subset = train_df.iloc[train_idx].copy()
val_subset = train_df.iloc[val_idx].copy()

# Build DataOps graph for the validation subset using the same preprocessing path
data = skrub.var("data", train_subset)
X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

# Keep preprocessing intact; focus improvements on the supervised model block
vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

# Small bounded comparison of stronger HGB configurations
model_candidates = [
    ("base", HistGradientBoostingRegressor(random_state=0)),
    (
        "deeper",
        HistGradientBoostingRegressor(
            random_state=0,
            learning_rate=0.05,
            max_depth=8,
            max_leaf_nodes=63,
            min_samples_leaf=20,
        ),
    ),
    (
        "regularized",
        HistGradientBoostingRegressor(
            random_state=0,
            learning_rate=0.03,
            max_depth=10,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            l2_regularization=0.1,
        ),
    ),
]

best_validation_score = np.inf
best_variant = None
best_learner = None

for variant_name, model in model_candidates:
    pred_graph = X_vec.skb.apply(model, y=y)
    learner = pred_graph.skb.make_learner(fitted=True)

    val_features = val_subset.drop(columns=[target_col], errors="ignore")
    val_pred = learner.predict({"data": val_features})
    validation_score = mean_squared_error(val_subset[target_col], val_pred) ** 0.5

    print(f"Ablation[{variant_name}] RMSE: {validation_score}")

    if validation_score < best_validation_score:
        best_validation_score = validation_score
        best_variant = variant_name
        best_learner = learner

final_validation_score = best_validation_score
print(f"Best ablation variant: {best_variant} | RMSE: {best_validation_score}")
print(f"Final Validation Performance: {final_validation_score}")

# Refit on the full training data with the best fixed configuration
full_data = skrub.var("data", train_df)
full_X = full_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
full_y = full_data[target_col].skb.mark_as_y()
full_X_vec = full_X.skb.apply(vectorizer)

best_model_params = {
    "base": dict(random_state=0),
    "deeper": dict(
        random_state=0,
        learning_rate=0.05,
        max_depth=8,
        max_leaf_nodes=63,
        min_samples_leaf=20,
    ),
    "regularized": dict(
        random_state=0,
        learning_rate=0.03,
        max_depth=10,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.1,
    ),
}[best_variant]

final_model = HistGradientBoostingRegressor(**best_model_params)
full_pred_graph = full_X_vec.skb.apply(final_model, y=full_y)
full_learner = full_pred_graph.skb.make_learner(fitted=True)

# Predict on test
test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
