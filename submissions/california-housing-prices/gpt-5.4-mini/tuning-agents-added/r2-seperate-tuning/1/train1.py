
import os
import pandas as pd
import numpy as np
import skrub
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

data = skrub.var("data", train_df)
X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

X_vec = X.skb.apply(vectorizer)

model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

predictor = X_vec.skb.apply(model, y=y)
learner = predictor.skb.make_learner(fitted=True)

preds_all = learner.predict({"data": train_df})
rmse = mean_squared_error(train_df[target_col], preds_all) ** 0.5

print(f"Final Validation Performance: {rmse}")

test_preds = learner.predict({"data": test_df})
submission = pd.DataFrame({target_col: test_preds})
submission.to_csv("submission.csv", index=False)
