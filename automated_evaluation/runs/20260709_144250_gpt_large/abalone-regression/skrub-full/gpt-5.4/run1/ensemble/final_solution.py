

import os
import glob
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression

random_state = 42
target_col = "Rings"


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0, None)
    return mean_squared_log_error(y_true, y_pred) ** 0.5


def find_csv(name):
    candidates = [
        os.path.join(".", "input", name),
        os.path.join(".", "input", "*", name),
    ]
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name} under ./input")


def build_predictor(train_subset):
    data_train = skrub.var("data", train_subset)

    X_train = (
        data_train
        .drop(columns=[target_col], errors="ignore")
        .skb.mark_as_X()
        .skb.apply(skrub.DropCols(cols=["id"]))
    )
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=random_state,
        verbose=-1
    )

    predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = predictor.skb.make_learner(fitted=True)
    return learner


train_path = find_csv("train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

n_train = len(train_part)
subsample_frac = 0.93
subsample_size = max(1, int(round(n_train * subsample_frac)))

rng2 = np.random.RandomState(random_state + 1)
rng3 = np.random.RandomState(random_state + 2)

sub_idx_2 = rng2.choice(n_train, size=subsample_size, replace=False)
sub_idx_3 = rng3.choice(n_train, size=subsample_size, replace=False)

train_part_2 = train_part.iloc[sub_idx_2].copy()
train_part_3 = train_part.iloc[sub_idx_3].copy()

learner_1 = build_predictor(train_part)
learner_2 = build_predictor(train_part_2)
learner_3 = build_predictor(train_part_3)

pred_1 = np.clip(np.asarray(learner_1.predict({"data": valid_part}), dtype=float).ravel(), 0, None)
pred_2 = np.clip(np.asarray(learner_2.predict({"data": valid_part}), dtype=float).ravel(), 0, None)
pred_3 = np.clip(np.asarray(learner_3.predict({"data": valid_part}), dtype=float).ravel(), 0, None)

y_valid = valid_part[target_col].to_numpy()

z1 = np.log1p(pred_1)
z2 = np.log1p(pred_2)
z3 = np.log1p(pred_3)
Z = np.column_stack([z1, z2, z3])
y_log = np.log1p(y_valid)

combiner = LinearRegression(fit_intercept=False)
combiner.fit(Z, y_log)

weights = np.asarray(combiner.coef_, dtype=float)
weights = np.clip(weights, 0, None)
if weights.sum() <= 0:
    leg_scores = np.array([
        rmsle(y_valid, pred_1),
        rmsle(y_valid, pred_2),
        rmsle(y_valid, pred_3),
    ], dtype=float)
    raw_weights = 1.0 / np.maximum(leg_scores, 1e-12) ** 2
    weights = raw_weights / raw_weights.sum()
else:
    weights = weights / weights.sum()

stack_log = Z @ weights
stack_pred = np.expm1(stack_log)
stack_pred = np.clip(stack_pred, 0, None)

best_score = None
best_pred = None
for alpha in [0.5, 0.7, 0.85]:
    final_pred = alpha * stack_pred + (1.0 - alpha) * pred_1
    final_pred = np.clip(final_pred, 0, None)
    score = rmsle(y_valid, final_pred)
    if best_score is None or score < best_score:
        best_score = score
        best_pred = final_pred

final_validation_score = rmsle(y_valid, best_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_path = find_csv("test.csv")
test_df = pd.read_csv(test_path)

n_full = len(train_df)
full_subsample_size = max(1, int(round(n_full * subsample_frac)))

full_sub_idx_2 = rng2.choice(n_full, size=full_subsample_size, replace=False)
full_sub_idx_3 = rng3.choice(n_full, size=full_subsample_size, replace=False)

train_df_2 = train_df.iloc[full_sub_idx_2].copy()
train_df_3 = train_df.iloc[full_sub_idx_3].copy()

full_learner_1 = build_predictor(train_df)
full_learner_2 = build_predictor(train_df_2)
full_learner_3 = build_predictor(train_df_3)

test_pred_1 = np.clip(np.asarray(full_learner_1.predict({"data": test_df}), dtype=float).ravel(), 0, None)
test_pred_2 = np.clip(np.asarray(full_learner_2.predict({"data": test_df}), dtype=float).ravel(), 0, None)
test_pred_3 = np.clip(np.asarray(full_learner_3.predict({"data": test_df}), dtype=float).ravel(), 0, None)

test_Z = np.column_stack([
    np.log1p(test_pred_1),
    np.log1p(test_pred_2),
    np.log1p(test_pred_3),
])

test_stack_log = test_Z @ weights
test_stack_pred = np.expm1(test_stack_log)
test_stack_pred = np.clip(test_stack_pred, 0, None)

best_alpha = None
best_score = None
for alpha in [0.5, 0.7, 0.85]:
    final_pred = alpha * stack_pred + (1.0 - alpha) * pred_1
    final_pred = np.clip(final_pred, 0, None)
    score = rmsle(y_valid, final_pred)
    if best_score is None or score < best_score:
        best_score = score
        best_alpha = alpha

test_final_pred = best_alpha * test_stack_pred + (1.0 - best_alpha) * test_pred_1
test_final_pred = np.clip(test_final_pred, 0, None)

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({
    "id": test_df["id"],
    "Rings": test_final_pred,
})
submission.to_csv("./final/submission.csv", index=False)
