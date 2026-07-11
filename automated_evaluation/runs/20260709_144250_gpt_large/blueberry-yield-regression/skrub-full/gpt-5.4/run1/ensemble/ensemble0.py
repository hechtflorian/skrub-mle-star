
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

# Pattern A: keep both legs intact as separate DataOps chains
lgb_predictor = X_train.skb.apply(vectorizer_lgb).skb.apply(lgb_model, y=y_train)
cat_predictor = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

lgb_learner = lgb_predictor.skb.make_learner(fitted=True)
cat_learner = cat_predictor.skb.make_learner(fitted=True)

valid_pred_lgb = np.asarray(lgb_learner.predict({"data": valid_part}), dtype=float).ravel()
valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part}), dtype=float).ravel()
y_valid = valid_part[target_col].to_numpy()

# Anchor blend
anchor_w_lgb = 0.50
anchor_w_cat = 0.50
anchor_pred = anchor_w_lgb * valid_pred_lgb + anchor_w_cat * valid_pred_cat
anchor_mae = mean_absolute_error(y_valid, anchor_pred)

# Conservative fixed-grid blend selection
weight_grid = [
    (1.00, 0.00),
    (0.75, 0.25),
    (0.67, 0.33),
    (0.50, 0.50),
    (0.33, 0.67),
    (0.25, 0.75),
    (0.00, 1.00),
]

best_w_lgb, best_w_cat = anchor_w_lgb, anchor_w_cat
best_mae = anchor_mae
best_pred = anchor_pred

for w_lgb, w_cat in weight_grid:
    blend_pred = w_lgb * valid_pred_lgb + w_cat * valid_pred_cat
    blend_mae = mean_absolute_error(y_valid, blend_pred)
    if blend_mae < best_mae:
        best_mae = blend_mae
        best_w_lgb, best_w_cat = w_lgb, w_cat
        best_pred = blend_pred

# Only switch away from 50/50 if improvement is clearly positive
improvement_threshold = 1e-4
if anchor_mae - best_mae > improvement_threshold:
    valid_pred = best_pred
else:
    best_w_lgb, best_w_cat = anchor_w_lgb, anchor_w_cat
    valid_pred = anchor_pred
    best_mae = anchor_mae

final_validation_score = mean_absolute_error(y_valid, valid_pred)
print(f"Selected blend weights -> LightGBM: {best_w_lgb:.2f}, CatBoost: {best_w_cat:.2f}")
print(f"Final Validation Performance: {final_validation_score}")
