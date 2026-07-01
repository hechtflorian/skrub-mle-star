
import os
import pandas as pd
import numpy as np
import skrub
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

vectorizer = skrub.TableVectorizer()
X_vec = X.skb.apply(vectorizer)

model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

pred_graph = X_vec.skb.apply(model, y=y)

train_idx = np.arange(len(train_df))
rng = np.random.RandomState(42)
rng.shuffle(train_idx)
val_size = max(1, int(0.2 * len(train_df)))
val_idx = train_idx[:val_size]
tr_idx = train_idx[val_size:]

X_tr_df = train_df.iloc[tr_idx].copy()
X_val_df = train_df.iloc[val_idx].copy()
y_tr = X_tr_df[target_col].copy()
y_val = X_val_df[target_col].copy()

X_tr = X_tr_df.drop(columns=[target_col])
X_val = X_val_df.drop(columns=[target_col])

vectorizer_fit = skrub.TableVectorizer()
X_tr_vec = vectorizer_fit.fit_transform(X_tr)
X_val_vec = vectorizer_fit.transform(X_val)

eval_model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
eval_model.fit(X_tr_vec, y_tr)
val_pred = eval_model.predict(X_val_vec)
final_validation_score = mean_squared_error(y_val, val_pred) ** 0.5

print(f"Final Validation Performance: {final_validation_score}")

full_model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
X_all = train_df.drop(columns=[target_col])
X_all_vec = vectorizer_fit.fit_transform(X_all)
full_model.fit(X_all_vec, train_df[target_col])
X_test_vec = vectorizer_fit.transform(test_df)
test_pred = full_model.predict(X_test_vec)

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
