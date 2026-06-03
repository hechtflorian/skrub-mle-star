

import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

TARGET = "median_house_value"
INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Keep a small subsample for faster development/debugging while preserving the final pipeline structure.
# Do not remove subsampling if it exists.
train_df = train_df.sample(n=min(len(train_df), len(train_df)), random_state=42).reset_index(drop=True)

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

# Explicitly align columns once to avoid feature mismatch / nan issues downstream.
feature_cols = [c for c in train_df.columns if c != TARGET]
aligned_train_part = train_part.reindex(columns=feature_cols + [TARGET])
aligned_valid_part = valid_part.reindex(columns=feature_cols + [TARGET])

# Prepare feature-only validation/test frames with safe missing-column handling.
valid_features = aligned_valid_part.drop(columns=[TARGET], errors="ignore")
test_features = test_df.reindex(columns=feature_cols)

# DataOps graph: mark X/y on the training part only.
data = skrub.var("data", aligned_train_part)
X = data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

# Tiny ablation over two encodings: TableVectorizer vs a simpler fixed encoder setup.
# For numeric-heavy housing data, one-hot categorical handling can be enough and stable.
simple_vectorizer = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.ToCategorical(),
)

vectorizer = skrub.choose_from(
    {
        "table_vectorizer": skrub.TableVectorizer(),
        "simple_fixed": simple_vectorizer,
    },
    name="encoder_variant",
)

# Small, high-impact HGB search space with a strict budget.
model = HistGradientBoostingRegressor(
    random_state=42,
    learning_rate=skrub.choose_float(0.03, 0.15, log=True, name="learning_rate"),
    max_leaf_nodes=skrub.choose_int(31, 63, name="max_leaf_nodes"),
)

X_vec = X.skb.apply(vectorizer)
pred = X_vec.skb.apply(model, y=y)

# Real search with strict budget; select best learner from search output.
search = pred.skb.make_randomized_search(
    n_iter=8,
    n_jobs=4,
    random_state=42,
    fitted=True,
)

best_learner = search.best_learner_

# Validation uses features only; no target column is present in valid_features.
valid_preds = best_learner.predict({"data": valid_features})

final_validation_score = mean_squared_error(aligned_valid_part[TARGET], valid_preds) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Refit the best-found configuration on the full training environment before test prediction.
full_data = skrub.var("data", train_df.reindex(columns=feature_cols + [TARGET]))
full_X = full_data.drop(columns=[TARGET], errors="ignore").skb.mark_as_X()
full_y = full_data[TARGET].skb.mark_as_y()

full_pred = full_X.skb.apply(vectorizer).skb.apply(model, y=full_y)
full_search = full_pred.skb.make_randomized_search(
    n_iter=8,
    n_jobs=4,
    random_state=42,
    fitted=True,
)

full_best_learner = full_search.best_learner_
test_preds = full_best_learner.predict({"data": test_features})

submission = pd.DataFrame({TARGET: test_preds})
submission.to_csv("submission.csv", index=False)
