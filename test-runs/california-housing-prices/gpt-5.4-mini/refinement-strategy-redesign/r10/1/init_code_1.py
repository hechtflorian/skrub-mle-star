
import os
import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

data = skrub.var("data", train_df)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

X_vec = X.skb.apply(vectorizer)

model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

pred = X_vec.skb.apply(model, y=y)

X_train_df, X_valid_df, y_train_df, y_valid_df = train_test_split(
    train_df.drop(columns=[target_col]),
    train_df[target_col],
    test_size=0.2,
    random_state=42,
)

train_env = {"data": pd.concat([X_train_df, y_train_df], axis=1)}
valid_env = {"data": pd.concat([X_valid_df, y_valid_df], axis=1)}

learner = pred.skb.make_learner(fitted=True)
learner.fit(train_env)

valid_pred = learner.predict(valid_env)
final_validation_score = mean_squared_error(y_valid_df, valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

test_pred = learner.predict({"data": test_df})
submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
