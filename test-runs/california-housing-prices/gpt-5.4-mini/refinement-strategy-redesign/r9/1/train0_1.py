
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"

data = skrub.var("data", train_df)
X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data[target_col].skb.mark_as_y()

# DataOps preprocessing path
vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

# Main model described in the prompt
cat_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    verbose=0,
)

# Additional reference model
lgbm_model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

# Keep DataOps graphs for both training paths
cat_predictor = X_vec.skb.apply(cat_model, y=y)
lgbm_predictor = X_vec.skb.apply(lgbm_model, y=y)

# Hold-out validation split
rng = np.random.RandomState(42)
indices = np.arange(len(train_df))
rng.shuffle(indices)
val_size = max(1, int(0.2 * len(train_df)))
val_idx = indices[:val_size]
tr_idx = indices[val_size:]

train_part = train_df.iloc[tr_idx].copy()
val_part = train_df.iloc[val_idx].copy()

X_tr = train_part.drop(columns=[target_col], errors="ignore")
y_tr = train_part[target_col].copy()
X_val = val_part.drop(columns=[target_col], errors="ignore")
y_val = val_part[target_col].copy()

# Fit preprocessing on the training split
vectorizer_fit = skrub.TableVectorizer()
X_tr_vec = vectorizer_fit.fit_transform(X_tr)
X_val_vec = vectorizer_fit.transform(X_val)

# Evaluate CatBoost
eval_cat = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    verbose=0,
)
eval_cat.fit(X_tr_vec, y_tr)
cat_val_pred = eval_cat.predict(X_val_vec)
cat_rmse = mean_squared_error(y_val, cat_val_pred) ** 0.5

# Evaluate LightGBM
eval_lgbm = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
eval_lgbm.fit(X_tr_vec, y_tr)
lgbm_val_pred = eval_lgbm.predict(X_val_vec)
lgbm_rmse = mean_squared_error(y_val, lgbm_val_pred) ** 0.5

# Simple ensemble on validation
val_pred = 0.5 * cat_val_pred + 0.5 * lgbm_val_pred
final_validation_score = mean_squared_error(y_val, val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Train final models on all training data
X_all = train_df.drop(columns=[target_col], errors="ignore")
y_all = train_df[target_col].copy()

vectorizer_final = skrub.TableVectorizer()
X_all_vec = vectorizer_final.fit_transform(X_all)

final_cat = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    verbose=0,
)
final_cat.fit(X_all_vec, y_all)

final_lgbm = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
final_lgbm.fit(X_all_vec, y_all)

X_test = test_df.copy()
X_test_vec = vectorizer_final.transform(X_test)

cat_test_pred = final_cat.predict(X_test_vec)
lgbm_test_pred = final_lgbm.predict(X_test_vec)
test_pred = 0.5 * cat_test_pred + 0.5 * lgbm_test_pred

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
