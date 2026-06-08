
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
# Base learner 1
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

# -------------------------
# Base learner 2
# -------------------------
# Keep the second pipeline separate and end-to-end, with the same preprocessing
# structure but a slightly different CatBoost configuration to provide diversity.
data2 = skrub.var("data2", train_part)
X2 = data2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y2 = data2[target_col].skb.mark_as_y()

X2 = X2.skb.apply(skrub.DropCols(cols=["households"]))

vectorizer2 = skrub.TableVectorizer()
X2_vec = X2.skb.apply(vectorizer2)

model2 = CatBoostRegressor(
    depth=10,
    learning_rate=0.03,
    iterations=5000,
    loss_function="RMSE",
    random_seed=123,
    verbose=0,
)

pred2 = X2_vec.skb.apply(model2, y=y2)
learner2 = pred2.skb.make_learner(fitted=True)
learner2.fit({"data2": train_part})

# -------------------------
# Validation predictions
# -------------------------
valid_pred1 = learner1.predict({"data1": valid_part})
valid_pred2 = learner2.predict({"data2": valid_part})

# Small merge layer: tune a weighted average on the validation split.
best_rmse = float("inf")
best_w = 0.5
best_valid_pred = None

for i in range(11):
    w2 = i / 10.0
    w1 = 1.0 - w2
    blended_valid = w1 * valid_pred1 + w2 * valid_pred2
    rmse = mean_squared_error(valid_part[target_col], blended_valid) ** 0.5
    if rmse < best_rmse:
        best_rmse = rmse
        best_w = w2
        best_valid_pred = blended_valid

final_validation_score = best_rmse
print(f"Final Validation Performance: {final_validation_score}")

# -------------------------
# Test predictions
# -------------------------
test_pred1 = learner1.predict({"data1": test_df})
test_pred2 = learner2.predict({"data2": test_df})

final_test_pred = (1.0 - best_w) * test_pred1 + best_w * test_pred2

submission = pd.DataFrame({"median_house_value": final_test_pred})
submission.to_csv("submission.csv", index=False)
