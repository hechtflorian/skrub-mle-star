
import os
import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"
TARGET_COL = "median_house_value"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

if len(train_df) > 20000:
    train_df = train_df.sample(n=20000, random_state=42).reset_index(drop=True)

rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_split = train_df.iloc[tr_idx].reset_index(drop=True)
valid_split = train_df.iloc[va_idx].reset_index(drop=True)

data = skrub.var("data", train_split)
X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
y = data[TARGET_COL].skb.mark_as_y()

vectorizer1 = skrub.TableVectorizer()
model1 = LGBMRegressor(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

pred_graph1 = X.skb.apply(vectorizer1).skb.apply(model1, y=y)
learner1 = pred_graph1.skb.make_learner(fitted=True)
learner1.fit({"data": train_split})
valid_pred1 = learner1.predict({"data": valid_split})

data2 = skrub.var("data", train_split)
X2 = data2.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
y2 = data2[TARGET_COL].skb.mark_as_y()

vectorizer2 = skrub.TableVectorizer()
model2 = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=300,
    random_state=42,
)

pred_graph2 = X2.skb.apply(vectorizer2).skb.apply(model2, y=y2)
learner2 = pred_graph2.skb.make_learner(fitted=True)
learner2.fit({"data": train_split})
valid_pred2 = learner2.predict({"data": valid_split})

y_valid = valid_split[TARGET_COL].to_numpy()
rmse1 = mean_squared_error(y_valid, valid_pred1) ** 0.5
rmse2 = mean_squared_error(y_valid, valid_pred2) ** 0.5

# Residual-correction ensemble on the held-out validation split.
if abs(rmse1 - rmse2) / max(rmse1, rmse2, 1e-12) < 0.01:
    # If models are very close, keep it simple and stable.
    if rmse1 <= rmse2:
        valid_pred = 0.7 * valid_pred1 + 0.3 * valid_pred2
        base_name = "LGBM"
    else:
        valid_pred = 0.7 * valid_pred2 + 0.3 * valid_pred1
        base_name = "HGB"
else:
    if rmse1 <= rmse2:
        base_pred = np.asarray(valid_pred1)
        aux_pred = np.asarray(valid_pred2)
        base_name = "LGBM"
    else:
        base_pred = np.asarray(valid_pred2)
        aux_pred = np.asarray(valid_pred1)
        base_name = "HGB"

    base_mean = float(np.mean(base_pred))
    aux_mean = float(np.mean(aux_pred))
    centered_base = base_pred - base_mean
    centered_aux = aux_pred - aux_mean

    meta_X = np.column_stack([centered_base, centered_aux])
    meta_model = Ridge(alpha=10.0, random_state=42)
    meta_model.fit(meta_X, y_valid - float(np.mean(y_valid)))

    meta_valid = float(np.mean(y_valid)) + meta_model.predict(meta_X)
    valid_pred = 0.8 * base_pred + 0.2 * meta_valid

final_validation_score = mean_squared_error(y_valid, valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

full_data1 = skrub.var("data", train_df)
X_full1 = full_data1.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
y_full1 = full_data1[TARGET_COL].skb.mark_as_y()

full_graph1 = X_full1.skb.apply(skrub.TableVectorizer()).skb.apply(
    LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    ),
    y=y_full1,
)
full_learner1 = full_graph1.skb.make_learner(fitted=True)
full_learner1.fit({"data": train_df})
test_pred1 = full_learner1.predict({"data": test_df})

full_data2 = skrub.var("data", train_df)
X_full2 = full_data2.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
y_full2 = full_data2[TARGET_COL].skb.mark_as_y()

full_graph2 = X_full2.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=42,
    ),
    y=y_full2,
)
full_learner2 = full_graph2.skb.make_learner(fitted=True)
full_learner2.fit({"data": train_df})
test_pred2 = full_learner2.predict({"data": test_df})

# Apply the same residual-correction logic on the test set using validation-derived centering.
if abs(rmse1 - rmse2) / max(rmse1, rmse2, 1e-12) < 0.01:
    if rmse1 <= rmse2:
        test_pred = 0.7 * test_pred1 + 0.3 * test_pred2
    else:
        test_pred = 0.7 * test_pred2 + 0.3 * test_pred1
else:
    if rmse1 <= rmse2:
        base_test = np.asarray(test_pred1)
        aux_test = np.asarray(test_pred2)
        base_mean = float(np.mean(valid_pred1))
        aux_mean = float(np.mean(valid_pred2))
    else:
        base_test = np.asarray(test_pred2)
        aux_test = np.asarray(test_pred1)
        base_mean = float(np.mean(valid_pred2))
        aux_mean = float(np.mean(valid_pred1))

    centered_base_test = base_test - base_mean
    centered_aux_test = aux_test - aux_mean
    test_meta_X = np.column_stack([centered_base_test, centered_aux_test])
    test_meta_pred = float(np.mean(y_valid)) + meta_model.predict(test_meta_X)
    test_pred = 0.8 * base_test + 0.2 * test_meta_pred

submission = pd.DataFrame({TARGET_COL: test_pred})
submission.to_csv("submission.csv", index=False)
