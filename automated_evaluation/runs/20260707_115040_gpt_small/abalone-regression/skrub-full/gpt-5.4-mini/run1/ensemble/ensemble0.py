
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")

target_col = "Rings"
random_state = 42

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(
        np.sqrt(
            np.mean(
                (
                    np.log1p(np.maximum(y_pred, 0))
                    - np.log1p(np.maximum(y_true, 0))
                ) ** 2
            )
        )
    )

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgbm_model = LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

cat_model = CatBoostRegressor(
    loss_function="RMSE",
    depth=8,
    learning_rate=0.05,
    iterations=3000,
    random_seed=random_state,
    verbose=0,
)

pred_lgbm = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)
pred_cat = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_cat.skb.make_learner(fitted=True)

valid_pred_lgbm = np.asarray(learner_lgbm.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_cat = np.asarray(learner_cat.predict({"data": valid_part}), dtype=float).ravel()

valid_pred_lgbm = np.maximum(valid_pred_lgbm, 0)
valid_pred_cat = np.maximum(valid_pred_cat, 0)

best_w = 0.6
best_score = float("inf")

for w in np.linspace(0.0, 1.0, 11):
    valid_pred = w * valid_pred_lgbm + (1.0 - w) * valid_pred_cat
    score = rmsle(valid_part[target_col].to_numpy(), valid_pred)
    if score < best_score:
        best_score = score
        best_w = float(w)

valid_pred = best_w * valid_pred_lgbm + (1.0 - best_w) * valid_pred_cat
final_validation_score = rmsle(valid_part[target_col].to_numpy(), valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
