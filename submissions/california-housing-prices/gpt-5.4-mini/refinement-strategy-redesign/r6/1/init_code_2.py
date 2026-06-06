
import pandas as pd
import numpy as np
import skrub
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

TARGET = "median_house_value"
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

data = skrub.var("data", train_part)
X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
y = data[TARGET].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

model = lgb.LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

pred_graph = X_vec.skb.apply(model, y=y)
learner = pred_graph.skb.make_learner(fitted=True)

valid_pred = learner.predict({"data": valid_part})
rmse = mean_squared_error(valid_part[TARGET], valid_pred) ** 0.5
print(f"Final Validation Performance: {rmse}")

full_data = skrub.var("data", train_df)
full_X = full_data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
full_y = full_data[TARGET].skb.mark_as_y()
full_X_vec = full_X.skb.apply(vectorizer)

full_pred_graph = full_X_vec.skb.apply(
    lgb.LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
    y=full_y,
)
full_learner = full_pred_graph.skb.make_learner(fitted=True)

test_pred = full_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
