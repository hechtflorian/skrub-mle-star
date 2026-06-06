
import os
import pandas as pd
import numpy as np
import skrub
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

TARGET = "median_house_value"
TRAIN_PATH = "./input/train.csv"
TEST_PATH = "./input/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Hold-out validation split
train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

def fit_predict_variant(train_df_local, valid_df_local, test_df_local, model_seed, split_seed=42):
    # DataOps-first pipeline
    data = skrub.var("data", train_df_local)
    X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=3000,
        random_seed=model_seed,
        verbose=200,
    )

    pred_graph = X_vec.skb.apply(model, y=y)
    learner = pred_graph.skb.make_learner(fitted=True)

    valid_pred = learner.predict({"data": valid_df_local})
    test_pred = learner.predict({"data": test_df_local})

    rmse = mean_squared_error(valid_df_local[TARGET], valid_pred) ** 0.5
    return valid_pred, test_pred, rmse

# Variant 1: original seed
valid_pred_1, test_pred_1, rmse_1 = fit_predict_variant(train_part, valid_part, test_df, model_seed=42)

# Variant 2: prediction-only variant with different CatBoost seed
valid_pred_2, test_pred_2, rmse_2 = fit_predict_variant(train_part, valid_part, test_df, model_seed=2024)

# Row-wise gate: winner per validation row based on smaller absolute residual
resid_1 = np.abs(valid_part[TARGET].to_numpy() - np.asarray(valid_pred_1))
resid_2 = np.abs(valid_part[TARGET].to_numpy() - np.asarray(valid_pred_2))
model1_wins = resid_1 < resid_2
p = float(np.mean(model1_wins))

final_weight_1 = float(np.clip(p, 0.2, 0.8))
final_weight_2 = 1.0 - final_weight_1

# Safety step: if one model is clearly better, use it alone
rmse_margin_threshold = 0.001
if rmse_1 + rmse_margin_threshold < rmse_2:
    final_test_pred = np.asarray(test_pred_1)
elif rmse_2 + rmse_margin_threshold < rmse_1:
    final_test_pred = np.asarray(test_pred_2)
else:
    final_test_pred = final_weight_1 * np.asarray(test_pred_1) + final_weight_2 * np.asarray(test_pred_2)

# Ensemble validation prediction using same global weight
if rmse_1 + rmse_margin_threshold < rmse_2:
    final_valid_pred = np.asarray(valid_pred_1)
elif rmse_2 + rmse_margin_threshold < rmse_1:
    final_valid_pred = np.asarray(valid_pred_2)
else:
    final_valid_pred = final_weight_1 * np.asarray(valid_pred_1) + final_weight_2 * np.asarray(valid_pred_2)

final_validation_score = mean_squared_error(valid_part[TARGET], final_valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Fit on full data for test prediction, keeping both DataOps flows intact
def fit_full_variant(train_df_full, test_df_local, model_seed):
    full_data = skrub.var("data", train_df_full)
    full_X = full_data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    full_y = full_data[TARGET].skb.mark_as_y()
    full_vectorizer = skrub.TableVectorizer()
    full_X_vec = full_X.skb.apply(full_vectorizer)

    full_model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=3000,
        random_seed=model_seed,
        verbose=200,
    )

    full_pred_graph = full_X_vec.skb.apply(full_model, y=full_y)
    full_learner = full_pred_graph.skb.make_learner(fitted=True)
    return full_learner.predict({"data": test_df_local})

test_pred_1_full = fit_full_variant(train_df, test_df, model_seed=42)
test_pred_2_full = fit_full_variant(train_df, test_df, model_seed=2024)

# Recompute blend on full-data predictions using validation-derived weights
if rmse_1 + rmse_margin_threshold < rmse_2:
    test_pred = np.asarray(test_pred_1_full)
elif rmse_2 + rmse_margin_threshold < rmse_1:
    test_pred = np.asarray(test_pred_2_full)
else:
    test_pred = final_weight_1 * np.asarray(test_pred_1_full) + final_weight_2 * np.asarray(test_pred_2_full)

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
