

import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "median_house_value"

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

data = skrub.var("data", train_part)
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
learner.fit({"data": train_part})

valid_pred = learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

test_pred = learner.predict({"data": test_df})
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
