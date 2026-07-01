
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

# -----------------------------
# Pipeline 1: original DataOps
# -----------------------------
data_1 = skrub.var("data", train_df)
X_1 = data_1.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_1 = data_1[target_col].skb.mark_as_y()

vectorizer_1 = skrub.TableVectorizer()
X_vec_1 = X_1.skb.apply(vectorizer_1)
model_1 = HistGradientBoostingRegressor(random_state=0)
pred_graph_1 = X_vec_1.skb.apply(model_1, y=y_1)
learner_1 = pred_graph_1.skb.make_learner(fitted=True)

# Holdout split
rng = np.random.RandomState(0)
indices = np.arange(len(train_df))
rng.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx, val_idx = indices[:split], indices[split:]

train_subset = train_df.iloc[train_idx].copy()
val_subset = train_df.iloc[val_idx].copy()

# Rebuild pipeline 1 on train subset
train_data_1 = skrub.var("data", train_subset)
train_X_1 = train_data_1.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
train_y_1 = train_data_1[target_col].skb.mark_as_y()
train_X_vec_1 = train_X_1.skb.apply(vectorizer_1)
train_pred_graph_1 = train_X_vec_1.skb.apply(model_1, y=train_y_1)
trained_learner_1 = train_pred_graph_1.skb.make_learner(fitted=True)

val_features_1 = val_subset.drop(columns=[target_col], errors="ignore")
val_pred_1 = trained_learner_1.predict({"data": val_features_1})

# Train on full data for test predictions
full_learner_1 = learner_1
pred_1_test = full_learner_1.predict({"data": test_df})

# -----------------------------
# Pipeline 2: separate full pipeline
# -----------------------------
data_2 = skrub.var("data", train_df)
X_2 = data_2.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_2 = data_2[target_col].skb.mark_as_y()

vectorizer_2 = skrub.TableVectorizer()
X_vec_2 = X_2.skb.apply(vectorizer_2)
model_2 = HistGradientBoostingRegressor(
    random_state=1,
    learning_rate=0.05,
    max_depth=6,
    max_iter=300
)
pred_graph_2 = X_vec_2.skb.apply(model_2, y=y_2)
learner_2 = pred_graph_2.skb.make_learner(fitted=True)

train_data_2 = skrub.var("data", train_subset)
train_X_2 = train_data_2.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
train_y_2 = train_data_2[target_col].skb.mark_as_y()
train_X_vec_2 = train_X_2.skb.apply(vectorizer_2)
train_pred_graph_2 = train_X_vec_2.skb.apply(model_2, y=train_y_2)
trained_learner_2 = train_pred_graph_2.skb.make_learner(fitted=True)

val_features_2 = val_subset.drop(columns=[target_col], errors="ignore")
val_pred_2 = trained_learner_2.predict({"data": val_features_2})

full_learner_2 = learner_2
pred_2_test = full_learner_2.predict({"data": test_df})

# -----------------------------
# Rank-aware piecewise ensemble
# -----------------------------
def rank01(a):
    a = np.asarray(a)
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    if len(a) > 1:
        ranks = ranks / (len(a) - 1)
    else:
        ranks = np.zeros_like(ranks, dtype=float)
    return ranks

rank_1 = rank01(pred_1_test)
rank_2 = rank01(pred_2_test)
rank_avg = 0.5 * rank_1 + 0.5 * rank_2

raw_mean = 0.5 * pred_1_test + 0.5 * pred_2_test
agreement = np.abs(pred_1_test - pred_2_test)
q20 = np.quantile(agreement, 0.2)
q80 = np.quantile(agreement, 0.8)

final_pred = raw_mean.copy()
low_mask = agreement <= q20
high_mask = agreement >= q80

final_pred[low_mask] = 0.7 * pred_1_test[low_mask] + 0.3 * pred_2_test[low_mask]
final_pred[high_mask] = 0.3 * pred_1_test[high_mask] + 0.7 * pred_2_test[high_mask]

# Optional tiny rank-aware adjustment: keep middle rows at median of pair
mid_mask = ~(low_mask | high_mask)
final_pred[mid_mask] = np.median(np.vstack([pred_1_test[mid_mask], pred_2_test[mid_mask]]), axis=0)

# Validation ensemble metric
val_rank_1 = rank01(val_pred_1)
val_rank_2 = rank01(val_pred_2)
val_agreement = np.abs(val_pred_1 - val_pred_2)
val_q20 = np.quantile(val_agreement, 0.2)
val_q80 = np.quantile(val_agreement, 0.8)

val_final_pred = 0.5 * val_pred_1 + 0.5 * val_pred_2
val_low_mask = val_agreement <= val_q20
val_high_mask = val_agreement >= val_q80
val_mid_mask = ~(val_low_mask | val_high_mask)

val_final_pred[val_low_mask] = 0.7 * val_pred_1[val_low_mask] + 0.3 * val_pred_2[val_low_mask]
val_final_pred[val_high_mask] = 0.3 * val_pred_1[val_high_mask] + 0.7 * val_pred_2[val_high_mask]
val_final_pred[val_mid_mask] = np.median(np.vstack([val_pred_1[val_mid_mask], val_pred_2[val_mid_mask]]), axis=0)

final_validation_score = mean_squared_error(val_subset[target_col], val_final_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({"median_house_value": final_pred})
submission.to_csv("submission.csv", index=False)
