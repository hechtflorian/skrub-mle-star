
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "yield"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


data_train = skrub.var("data", train_part)

drop_redundant_cols = ["id", "AverageOfUpperTRange", "AverageOfLowerTRange", "AverageRainingDays"]

X_train = (
    data_train
    .drop(columns=target_col, errors="ignore")
    .drop(columns=drop_redundant_cols, errors="ignore")
    .skb.mark_as_X()
)
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_lgb = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgb_model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="mae",
    random_state=random_state,
    verbose=-1,
)

cat_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=random_state,
    verbose=0,
)

lgb_predictor = X_train.skb.apply(vectorizer_lgb).skb.apply(lgb_model, y=y_train)
cat_predictor = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

lgb_learner = lgb_predictor.skb.make_learner(fitted=True)
cat_learner = cat_predictor.skb.make_learner(fitted=True)


valid_pred_lgb = np.asarray(lgb_learner.predict({"data": valid_part}), dtype=float)
valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part}), dtype=float)

valid_pred = 0.5 * valid_pred_lgb + 0.5 * valid_pred_cat

final_validation_score = mean_absolute_error(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
