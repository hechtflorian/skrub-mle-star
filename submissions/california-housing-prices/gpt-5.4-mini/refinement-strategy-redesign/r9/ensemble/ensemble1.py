
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
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

# Model described in the prompt
model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    verbose=0,
)

# Keep a DataOps graph for the main training path
predictor = X_vec.skb.apply(model, y=y)

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

# Fit preprocessing and model on the training split
vectorizer_fit = skrub.TableVectorizer()
X_tr_vec = vectorizer_fit.fit_transform(X_tr)
X_val_vec = vectorizer_fit.transform(X_val)

eval_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    verbose=0,
)
eval_model.fit(X_tr_vec, y_tr)

val_pred = eval_model.predict(X_val_vec)
final_validation_score = mean_squared_error(y_val, val_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# -----------------------------
# Ensemble training for test predictions
# Anchor + bootstrap bagging blend
# -----------------------------
X_all = train_df.drop(columns=[target_col], errors="ignore")
y_all = train_df[target_col].copy()

# Anchor model on full data
vectorizer_anchor = skrub.TableVectorizer()
X_all_vec = vectorizer_anchor.fit_transform(X_all)

anchor_model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=3000,
    learning_rate=0.03,
    depth=8,
    random_seed=42,
    verbose=0,
)
anchor_model.fit(X_all_vec, y_all)

X_test = test_df.copy()
X_test_vec_anchor = vectorizer_anchor.transform(X_test)
anchor_test_pred = anchor_model.predict(X_test_vec_anchor)

# Bootstrap members
n_bootstrap = 4
bootstrap_preds = []
bootstrap_weights = []

boot_rng = np.random.RandomState(42)

for b in range(n_bootstrap):
    sample_idx = boot_rng.randint(0, len(X_all), size=len(X_all))
    oob_mask = np.ones(len(X_all), dtype=bool)
    oob_mask[np.unique(sample_idx)] = False
    oob_idx = np.where(oob_mask)[0]

    X_boot = X_all.iloc[sample_idx].copy()
    y_boot = y_all.iloc[sample_idx].copy()

    vectorizer_boot = skrub.TableVectorizer()
    X_boot_vec = vectorizer_boot.fit_transform(X_boot)

    boot_model = CatBoostRegressor(
        loss_function="RMSE",
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        random_seed=42 + b + 1,
        verbose=0,
    )
    boot_model.fit(X_boot_vec, y_boot)

    X_test_vec_boot = vectorizer_boot.transform(X_test)
    boot_test_pred = boot_model.predict(X_test_vec_boot)
    bootstrap_preds.append(boot_test_pred)

    # Optional adaptive weight from OOB error
    if len(oob_idx) > 0:
        X_oob = X_all.iloc[oob_idx].copy()
        y_oob = y_all.iloc[oob_idx].copy()
        X_oob_vec = vectorizer_boot.transform(X_oob)
        oob_pred = boot_model.predict(X_oob_vec)
        oob_rmse = mean_squared_error(y_oob, oob_pred) ** 0.5
        bootstrap_weights.append(1.0 / (oob_rmse + 1e-6))
    else:
        bootstrap_weights.append(1.0)

bootstrap_preds = np.asarray(bootstrap_preds)

# Weighted blend: anchor gets largest share, bootstrap mean gets remaining share
anchor_weight = 0.65
bootstrap_total_weight = 1.0 - anchor_weight

bootstrap_weights = np.asarray(bootstrap_weights, dtype=float)
bootstrap_weights = bootstrap_weights / bootstrap_weights.sum()

bootstrap_blend = np.sum(bootstrap_preds * bootstrap_weights[:, None], axis=0)
test_pred = anchor_weight * anchor_test_pred + bootstrap_total_weight * bootstrap_blend

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
