
import os
import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
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

lgbm_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=6,
    random_seed=42,
    verbose=False,
)

lgbm_predictor = X_vec.skb.apply(lgbm_model, y=y)
cat_predictor = X_vec.skb.apply(cat_model, y=y)

rng = np.random.RandomState(42)
idx = rng.permutation(len(train_df))
val_size = int(len(train_df) * 0.2)
val_idx = idx[:val_size]
fit_idx = idx[val_size:]

fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

lgbm_learner = lgbm_predictor.skb.make_learner(fitted=True)
cat_learner = cat_predictor.skb.make_learner(fitted=True)

_ = lgbm_learner.predict({"data": fit_df})
_ = cat_learner.predict({"data": fit_df})

lgbm_val_pred = lgbm_learner.predict({"data": val_df})
cat_val_pred = cat_learner.predict({"data": val_df})

val_pred = 0.5 * lgbm_val_pred + 0.5 * cat_val_pred

y_true = val_df[target].values
rmse = mean_squared_error(y_true, val_pred) ** 0.5
print(f"Final Validation Performance: {rmse}")

final_lgbm_learner = lgbm_predictor.skb.make_learner(fitted=True)
final_cat_learner = cat_predictor.skb.make_learner(fitted=True)

_ = final_lgbm_learner.predict({"data": train_df})
_ = final_cat_learner.predict({"data": train_df})

lgbm_test_pred = final_lgbm_learner.predict({"data": test_df})
cat_test_pred = final_cat_learner.predict({"data": test_df})

test_pred = 0.5 * lgbm_test_pred + 0.5 * cat_test_pred

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
