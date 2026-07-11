
import os
import glob
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

random_state = 42
target_col = "revenue"

train_path = glob.glob(os.path.join("./input", "train.csv"))[0]
train_df = pd.read_csv(train_path)

def add_datetime_features(df):
    out = df.copy()
    if "Open Date" in out.columns:
        out["Open Date"] = pd.to_datetime(out["Open Date"], errors="coerce")
        out["Year"] = out["Open Date"].dt.year
        out["Month"] = out["Open Date"].dt.month
        out["Day"] = out["Open Date"].dt.day
        out["Age"] = 2026 - out["Year"]
    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_datetime_features)

X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_cat = skrub.TableVectorizer(
    low_cardinality="drop",
    high_cardinality="drop",
)

vectorizer_lgbm = skrub.TableVectorizer()

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    random_seed=random_state,
    verbose=0,
)

lgbm_model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)

pred_cat = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)
pred_lgbm = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)

learner_cat = pred_cat.skb.make_learner(fitted=True)
learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)

valid_pred_cat = learner_cat.predict({"data": valid_part})
valid_pred_lgbm = learner_lgbm.predict({"data": valid_part})

valid_pred = 0.5 * np.asarray(valid_pred_cat, dtype=float) + 0.5 * np.asarray(valid_pred_lgbm, dtype=float)

final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
