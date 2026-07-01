
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "median_house_value"

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

# -------------------------
# Pipeline 1 (unchanged)
# -------------------------
data1 = skrub.var("data1", train_part)
X1 = data1.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y1 = data1[target_col].skb.mark_as_y()

# Structural refinement from ablation: remove redundant households before vectorization.
X1 = X1.skb.apply(skrub.DropCols(cols=["households"]))

vectorizer1 = skrub.TableVectorizer()
X1_vec = X1.skb.apply(vectorizer1)

model1 = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=4000,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)

pred1 = X1_vec.skb.apply(model1, y=y1)
learner1 = pred1.skb.make_learner(fitted=True)
learner1.fit({"data1": train_part})

valid_pred1 = learner1.predict({"data1": valid_part})
test_pred1 = learner1.predict({"data1": test_df})

# -------------------------
# Pipeline 2 (kept fully intact in spirit; same core settings)
# -------------------------
data2 = skrub.var("data2", train_part)
X2 = data2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y2 = data2[target_col].skb.mark_as_y()

# Keep preprocessing/model settings unchanged except for independent instantiation.
X2 = X2.skb.apply(skrub.DropCols(cols=["households"]))

vectorizer2 = skrub.TableVectorizer()
X2_vec = X2.skb.apply(vectorizer2)

model2 = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=4000,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)

pred2 = X2_vec.skb.apply(model2, y=y2)
learner2 = pred2.skb.make_learner(fitted=True)
learner2.fit({"data2": train_part})

valid_pred2 = learner2.predict({"data2": valid_part})
test_pred2 = learner2.predict({"data2": test_df})

# -------------------------
# Dynamic bin-based gating ensemble
# -------------------------
valid_target = valid_part[target_col].values
abs_diff_valid = np.abs(valid_pred1 - valid_pred2)
abs_diff_test = np.abs(test_pred1 - test_pred2)

def grid_search_bin_weight(y_true, p1, p2, mask, candidate_weights):
    if mask.sum() == 0:
        return 0.5
    best_w = 0.5
    best_rmse = float("inf")
    for w in candidate_weights:
        blended = w * p1[mask] + (1.0 - w) * p2[mask]
        rmse = mean_squared_error(y_true[mask], blended) ** 0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w
    return best_w

# Coarse and safe: bottom 50% disagreement vs top 50% disagreement.
median_diff = np.median(abs_diff_valid)
low_mask = abs_diff_valid <= median_diff
high_mask = abs_diff_valid > median_diff

candidate_weights = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)

w_low = grid_search_bin_weight(valid_target, valid_pred1, valid_pred2, low_mask, candidate_weights)
w_high = grid_search_bin_weight(valid_target, valid_pred1, valid_pred2, high_mask, candidate_weights)

# Optional sanity fallback: if the two bin weights collapse to extremes or are identical, use average in low bin.
if not np.isfinite(w_low):
    w_low = 0.5
if not np.isfinite(w_high):
    w_high = 0.5

test_low_mask = abs_diff_test <= median_diff
test_high_mask = abs_diff_test > median_diff

final_valid_pred = np.empty_like(valid_pred1, dtype=float)
final_valid_pred[low_mask] = w_low * valid_pred1[low_mask] + (1.0 - w_low) * valid_pred2[low_mask]
final_valid_pred[high_mask] = w_high * valid_pred1[high_mask] + (1.0 - w_high) * valid_pred2[high_mask]

final_test_pred = np.empty_like(test_pred1, dtype=float)
final_test_pred[test_low_mask] = w_low * test_pred1[test_low_mask] + (1.0 - w_low) * test_pred2[test_low_mask]
final_test_pred[test_high_mask] = w_high * test_pred1[test_high_mask] + (1.0 - w_high) * test_pred2[test_high_mask]

final_validation_score = mean_squared_error(valid_target, final_valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({"median_house_value": final_test_pred})
submission.to_csv("submission.csv", index=False)
