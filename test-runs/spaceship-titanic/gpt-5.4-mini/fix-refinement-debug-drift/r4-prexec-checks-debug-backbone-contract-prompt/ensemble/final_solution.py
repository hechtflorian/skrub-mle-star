
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

os.makedirs("./final", exist_ok=True)

# Load data
train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "Transported"

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

# --- View 1: original CatBoost pipeline (kept intact) ---
vectorizer_1 = skrub.TableVectorizer()
pred_1 = X_train.skb.apply(vectorizer_1).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
    y=y_train,
)

learner_1 = pred_1.skb.make_learner(fitted=True)
valid_pred_1 = np.asarray(learner_1.predict({"data": valid_part})).ravel()

# --- View 2: same pipeline, tiny inference-safe perturbation via bootstrap resample ---
bootstrap_idx = np.random.RandomState(42).choice(
    len(train_part), size=len(train_part), replace=True
)
train_bootstrap = train_part.iloc[bootstrap_idx].copy()

data_boot = skrub.var("data", train_bootstrap)
X_boot = data_boot.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_boot = data_boot[target_col].skb.mark_as_y()

vectorizer_2 = skrub.TableVectorizer()
pred_2 = X_boot.skb.apply(vectorizer_2).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
    y=y_boot,
)

learner_2 = pred_2.skb.make_learner(fitted=True)
valid_pred_2 = np.asarray(learner_2.predict({"data": valid_part})).ravel()

# Discrete weight search on the validation split only
weight_candidates = [(0.8, 0.2), (0.7, 0.3), (0.6, 0.4)]
best_weight = weight_candidates[0]
best_score = -1.0

for w1, w2 in weight_candidates:
    valid_blend = w1 * valid_pred_1 + w2 * valid_pred_2
    valid_blend_bool = valid_blend > 0.5
    score = accuracy_score(valid_part[target_col].values, valid_blend_bool)
    if score > best_score:
        best_score = score
        best_weight = (w1, w2)

# Final validation performance with best weighted blend
w1, w2 = best_weight
valid_final_pred = w1 * valid_pred_1 + w2 * valid_pred_2
valid_final_pred_bool = valid_final_pred > 0.5
final_validation_score = accuracy_score(valid_part[target_col].values, valid_final_pred_bool)
print(f"Final Validation Performance: {final_validation_score}")

# Submission-stage refit on full training data
data_full_1 = skrub.var("data", train_df)
X_full_1 = data_full_1.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full_1 = data_full_1[target_col].skb.mark_as_y()

vectorizer_full_1 = skrub.TableVectorizer()
full_pred_1 = X_full_1.skb.apply(vectorizer_full_1).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
    y=y_full_1,
)

full_learner_1 = full_pred_1.skb.make_learner(fitted=True)
test_pred_1 = np.asarray(full_learner_1.predict({"data": test_df})).ravel()

# Bootstrap view on full training data
bootstrap_idx_full = np.random.RandomState(42).choice(
    len(train_df), size=len(train_df), replace=True
)
train_bootstrap_full = train_df.iloc[bootstrap_idx_full].copy()

data_full_2 = skrub.var("data", train_bootstrap_full)
X_full_2 = data_full_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full_2 = data_full_2[target_col].skb.mark_as_y()

vectorizer_full_2 = skrub.TableVectorizer()
full_pred_2 = X_full_2.skb.apply(vectorizer_full_2).skb.apply(
    CatBoostClassifier(
        loss_function="Logloss",
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
    y=y_full_2,
)

full_learner_2 = full_pred_2.skb.make_learner(fitted=True)
test_pred_2 = np.asarray(full_learner_2.predict({"data": test_df})).ravel()

test_pred = w1 * test_pred_1 + w2 * test_pred_2
test_pred_bool = test_pred > 0.5

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred_bool.astype(bool),
    }
)
submission["Transported"] = submission["Transported"].map({True: "True", False: "False"})
submission.to_csv("./final/submission.csv", index=False)
