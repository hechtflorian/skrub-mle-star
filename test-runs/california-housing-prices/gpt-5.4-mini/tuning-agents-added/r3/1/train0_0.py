
import os
import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target = "median_house_value"

data = skrub.var("data", train_df)
X = data.drop(columns=target, errors="ignore").skb.mark_as_X()
y = data[target].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

X_vec = X.skb.apply(vectorizer)

model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

predictor = X_vec.skb.apply(model, y=y)

train_idx = np.random.RandomState(42).permutation(len(train_df))
val_size = int(len(train_df) * 0.2)
val_idx = train_idx[:val_size]
fit_idx = train_idx[val_size:]

fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

fitted_learner = predictor.skb.make_learner(fitted=True)

_ = fitted_learner.predict({"data": fit_df})
val_pred = fitted_learner.predict({"data": val_df})

y_true = val_df[target].values
rmse = mean_squared_error(y_true, val_pred) ** 0.5
print(f"Final Validation Performance: {rmse}")

final_learner = predictor.skb.make_learner(fitted=True)
_ = final_learner.predict({"data": train_df})
test_pred = final_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
