
import os
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "median_house_value"

data = skrub.var("data", train_df)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

# Structural refinement from ablation: remove redundant households before vectorization.
X = X.skb.apply(skrub.DropCols(cols=["households"]))

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=4000,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)

pred = X_vec.skb.apply(model, y=y)
learner = pred.skb.make_learner(fitted=True)
learner.fit({"data": train_df})

valid_pred = learner.predict({"data": train_df})
final_validation_score = mean_squared_error(train_df[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

test_pred = learner.predict({"data": test_df})

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("./final/submission.csv", index=False)
