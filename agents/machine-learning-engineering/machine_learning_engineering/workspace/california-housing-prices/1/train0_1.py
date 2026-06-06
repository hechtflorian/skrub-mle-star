
import pandas as pd
import numpy as np
import skrub
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

target_col = "median_house_value"

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

data = skrub.var("data", train_part)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

X_vec = X.skb.apply(vectorizer)

cat_model = CatBoostRegressor(
    depth=8,
    learning_rate=0.05,
    iterations=4000,
    loss_function="RMSE",
    random_seed=42,
    verbose=0,
)

lgbm_model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=64,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

cat_pred = X_vec.skb.apply(cat_model, y=y)
lgbm_pred = X_vec.skb.apply(lgbm_model, y=y)

cat_learner = cat_pred.skb.make_learner(fitted=True)
lgbm_learner = lgbm_pred.skb.make_learner(fitted=True)

cat_learner.fit({"data": train_part})
lgbm_learner.fit({"data": train_part})

valid_cat_pred = cat_learner.predict({"data": valid_part})
valid_lgbm_pred = lgbm_learner.predict({"data": valid_part})

valid_pred = 0.5 * np.asarray(valid_cat_pred) + 0.5 * np.asarray(valid_lgbm_pred)
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

test_cat_pred = cat_learner.predict({"data": test_df})
test_lgbm_pred = lgbm_learner.predict({"data": test_df})
test_pred = 0.5 * np.asarray(test_cat_pred) + 0.5 * np.asarray(test_lgbm_pred)

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
