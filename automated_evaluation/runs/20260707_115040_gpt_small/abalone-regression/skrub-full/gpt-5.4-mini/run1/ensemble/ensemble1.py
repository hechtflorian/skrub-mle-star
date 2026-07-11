
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

log_lgbm = np.log1p(valid_pred_lgbm)
log_cat = np.log1p(valid_pred_cat)

best_score = np.inf
best_pred = None

for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    log_blend = w * log_lgbm + (1.0 - w) * log_cat
    blended = np.expm1(log_blend)
    blended = np.maximum(blended, 0)

    for alpha in [0.8, 1.0, 1.2]:
        pred = np.maximum(blended, 0) ** alpha
        score = rmsle(valid_part[target_col].to_numpy(), pred)
        if score < best_score:
            best_score = score
            best_pred = pred

median_pred = np.median(np.vstack([valid_pred_lgbm, valid_pred_cat]), axis=0)
median_pred = np.maximum(median_pred, 0)
median_score = rmsle(valid_part[target_col].to_numpy(), median_pred)

if median_score < best_score:
    final_validation_score = median_score
else:
    final_validation_score = best_score

print(f"Final Validation Performance: {final_validation_score}")
