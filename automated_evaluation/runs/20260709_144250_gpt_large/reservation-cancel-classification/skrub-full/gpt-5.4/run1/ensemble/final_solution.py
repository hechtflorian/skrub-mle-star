
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "booking_status"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def train_leg_and_predict(leg_df, valid_df):
    data_train = skrub.var("data", leg_df)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        random_state=random_state,
        verbose=0,
    )

    predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = predictor.skb.make_learner(fitted=True)
    pred = learner.predict_proba({"data": valid_df})[:, 1]
    return learner, pred


def safe_corr(a, b):
    corr = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(corr):
        corr = 0.0
    return float(np.clip(corr, -1.0, 1.0))


def rank_average(pred_list):
    pred_array = np.column_stack(pred_list)
    ranks = np.zeros_like(pred_array, dtype=float)
    for j in range(pred_array.shape[1]):
        order = np.argsort(pred_array[:, j], kind="mergesort")
        ranks[order, j] = np.arange(len(order), dtype=float)
    rank_avg = ranks.mean(axis=1)
    if len(rank_avg) > 1:
        rank_avg = rank_avg / (len(rank_avg) - 1)
    return rank_avg


n_train = len(train_part)
window_size = int(0.8 * n_train)

full_df = train_part
first80_df = train_part.iloc[:window_size].copy()
last80_df = train_part.iloc[n_train - window_size:].copy()
middle_start = (n_train - window_size) // 2
middle80_df = train_part.iloc[middle_start:middle_start + window_size].copy()

_, p_full = train_leg_and_predict(full_df, valid_part)
_, p_first = train_leg_and_predict(first80_df, valid_part)
_, p_last = train_leg_and_predict(last80_df, valid_part)
_, p_middle = train_leg_and_predict(middle80_df, valid_part)

corr_first = safe_corr(p_full, p_first)
corr_last = safe_corr(p_full, p_last)
corr_middle = safe_corr(p_full, p_middle)

diversities = np.array([
    max(1e-8, 1.0 - corr_first),
    max(1e-8, 1.0 - corr_last),
    max(1e-8, 1.0 - corr_middle),
], dtype=float)
subset_weights = 0.45 * diversities / diversities.sum()

weighted_raw_blend = (
    0.55 * p_full
    + subset_weights[0] * p_first
    + subset_weights[1] * p_last
    + subset_weights[2] * p_middle
)

subset_rank_avg = rank_average([p_first, p_last, p_middle])
anchor_rank_blend = 0.6 * p_full + 0.4 * subset_rank_avg

equal_avg_blend = (p_full + p_first + p_last + p_middle) / 4.0

y_valid = valid_part[target_col].values

score_weighted_raw = roc_auc_score(y_valid, weighted_raw_blend)
score_anchor_rank = roc_auc_score(y_valid, anchor_rank_blend)
score_equal_avg = roc_auc_score(y_valid, equal_avg_blend)

final_validation_score = max(
    score_weighted_raw,
    score_anchor_rank,
    score_equal_avg,
)

print(f"Final Validation Performance: {final_validation_score}")

import os

test_df = pd.read_csv("./input/test.csv")

n_train_full = len(train_df)
window_size_full = int(0.8 * n_train_full)

full_df_full = train_df
first80_df_full = train_df.iloc[:window_size_full].copy()
last80_df_full = train_df.iloc[n_train_full - window_size_full:].copy()
middle_start_full = (n_train_full - window_size_full) // 2
middle80_df_full = train_df.iloc[middle_start_full:middle_start_full + window_size_full].copy()

learner_full, test_p_full = train_leg_and_predict(full_df_full, test_df)
learner_first, test_p_first = train_leg_and_predict(first80_df_full, test_df)
learner_last, test_p_last = train_leg_and_predict(last80_df_full, test_df)
learner_middle, test_p_middle = train_leg_and_predict(middle80_df_full, test_df)

corr_first_full = safe_corr(test_p_full, test_p_first)
corr_last_full = safe_corr(test_p_full, test_p_last)
corr_middle_full = safe_corr(test_p_full, test_p_middle)

diversities_full = np.array([
    max(1e-8, 1.0 - corr_first_full),
    max(1e-8, 1.0 - corr_last_full),
    max(1e-8, 1.0 - corr_middle_full),
], dtype=float)
subset_weights_full = 0.45 * diversities_full / diversities_full.sum()

weighted_raw_blend_test = (
    0.55 * test_p_full
    + subset_weights_full[0] * test_p_first
    + subset_weights_full[1] * test_p_last
    + subset_weights_full[2] * test_p_middle
)

subset_rank_avg_test = rank_average([test_p_first, test_p_last, test_p_middle])
anchor_rank_blend_test = 0.6 * test_p_full + 0.4 * subset_rank_avg_test

equal_avg_blend_test = (
    test_p_full + test_p_first + test_p_last + test_p_middle
) / 4.0

if final_validation_score == score_weighted_raw:
    final_test_pred = weighted_raw_blend_test
elif final_validation_score == score_anchor_rank:
    final_test_pred = anchor_rank_blend_test
else:
    final_test_pred = equal_avg_blend_test

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({
    "id": test_df["id"],
    target_col: final_test_pred,
})
submission.to_csv("./final/submission.csv", index=False)
