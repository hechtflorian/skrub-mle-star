
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

# Common hold-out split
rng = np.random.RandomState(0)
indices = np.arange(len(train_df))
rng.shuffle(indices)
split = int(len(indices) * 0.8)
train_idx, val_idx = indices[:split], indices[split:]

train_subset = train_df.iloc[train_idx].copy()
val_subset = train_df.iloc[val_idx].copy()

# -------------------------------------------------
# Solution 1: original DataOps pipeline
# -------------------------------------------------
data1 = skrub.var("data", train_subset)
X1 = data1.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y1 = data1[target_col].skb.mark_as_y()

vectorizer1 = skrub.TableVectorizer()
X1_vec = X1.skb.apply(vectorizer1)

model1 = HistGradientBoostingRegressor(random_state=0)
pred_graph1 = X1_vec.skb.apply(model1, y=y1)
learner1 = pred_graph1.skb.make_learner(fitted=True)

# validation prediction for solution 1
val_features = val_subset.drop(columns=[target_col], errors="ignore")
pred_1_val = learner1.predict({"data": val_features})
rmse_1 = mean_squared_error(val_subset[target_col], pred_1_val) ** 0.5

# fit on full data and predict test for solution 1
data1_full = skrub.var("data", train_df)
X1_full = data1_full.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y1_full = data1_full[target_col].skb.mark_as_y()
X1_full_vec = X1_full.skb.apply(vectorizer1)
pred_graph1_full = X1_full_vec.skb.apply(model1, y=y1_full)
full_learner1 = pred_graph1_full.skb.make_learner(fitted=True)
pred_1_test = np.asarray(full_learner1.predict({"data": test_df}))

# -------------------------------------------------
# Solution 2: second DataOps pipeline (same workflow, different fixed model)
# -------------------------------------------------
data2 = skrub.var("data", train_subset)
X2 = data2.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y2 = data2[target_col].skb.mark_as_y()

vectorizer2 = skrub.TableVectorizer()
X2_vec = X2.skb.apply(vectorizer2)

model2 = HistGradientBoostingRegressor(
    random_state=1,
    max_depth=6,
    learning_rate=0.05,
    max_iter=300,
)
pred_graph2 = X2_vec.skb.apply(model2, y=y2)
learner2 = pred_graph2.skb.make_learner(fitted=True)

pred_2_val = learner2.predict({"data": val_features})
rmse_2 = mean_squared_error(val_subset[target_col], pred_2_val) ** 0.5

# fit on full data and predict test for solution 2
data2_full = skrub.var("data", train_df)
X2_full = data2_full.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y2_full = data2_full[target_col].skb.mark_as_y()
X2_full_vec = X2_full.skb.apply(vectorizer2)
pred_graph2_full = X2_full_vec.skb.apply(model2, y=y2_full)
full_learner2 = pred_graph2_full.skb.make_learner(fitted=True)
pred_2_test = np.asarray(full_learner2.predict({"data": test_df}))

# -------------------------------------------------
# Ensemble: inverse-RMSE weighted average
# -------------------------------------------------
eps = 1e-12
w1 = 1.0 / (rmse_1 + eps)
w2 = 1.0 / (rmse_2 + eps)
w_sum = w1 + w2
w1 /= w_sum
w2 /= w_sum

final_pred = w1 * pred_1_test + w2 * pred_2_test

# Optional light robustness against extreme outputs
median_blend = np.median(np.vstack([pred_1_test, pred_2_test]), axis=0)
final_pred = 0.5 * final_pred + 0.5 * median_blend

# Validation score of the blended model on hold-out
pred_1_val = np.asarray(pred_1_val)
pred_2_val = np.asarray(pred_2_val)
val_blend = w1 * pred_1_val + w2 * pred_2_val
val_blend = 0.5 * val_blend + 0.5 * np.median(np.vstack([pred_1_val, pred_2_val]), axis=0)
final_validation_score = mean_squared_error(val_subset[target_col], val_blend) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({"median_house_value": final_pred})
submission.to_csv("submission.csv", index=False)
