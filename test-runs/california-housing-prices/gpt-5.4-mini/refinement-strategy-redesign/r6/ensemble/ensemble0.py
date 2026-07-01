
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

def train_and_predict(train_data, valid_data, test_data, seed=42, depth=8, learning_rate=0.05, iterations=3000):
    data = skrub.var("data", train_data)
    X = data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    y = data[TARGET].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)

    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        random_seed=seed,
        verbose=200,
    )

    pred_graph = X_vec.skb.apply(model, y=y)
    learner = pred_graph.skb.make_learner(fitted=True)

    valid_pred = learner.predict({"data": valid_data})

    full_data = skrub.var("data", train_data)
    full_X = full_data.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
    full_y = full_data[TARGET].skb.mark_as_y()
    full_X_vec = full_X.skb.apply(vectorizer)

    full_pred_graph = full_X_vec.skb.apply(
        CatBoostRegressor(
            loss_function="RMSE",
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            random_seed=seed,
            verbose=200,
        ),
        y=full_y,
    )
    full_learner = full_pred_graph.skb.make_learner(fitted=True)
    test_pred = full_learner.predict({"data": test_data})

    return valid_pred, test_pred

# Base model 1: original solution as-is
valid_pred_1, test_pred_1 = train_and_predict(
    train_part, valid_part, test_df, seed=42, depth=8, learning_rate=0.05, iterations=3000
)
rmse_1 = mean_squared_error(valid_part[TARGET], valid_pred_1) ** 0.5
print(f"Ablation[base_model_1] RMSE: {rmse_1}")

# Base model 2: lightly perturbed clone of the same pipeline
valid_pred_2, test_pred_2 = train_and_predict(
    train_part, valid_part, test_df, seed=123, depth=7, learning_rate=0.06, iterations=2500
)
rmse_2 = mean_squared_error(valid_part[TARGET], valid_pred_2) ** 0.5
print(f"Ablation[base_model_2] RMSE: {rmse_2}")

# Prediction-layer ensemble only
eps = 1e-12
w1 = 1.0 / (rmse_1 + eps)
w2 = 1.0 / (rmse_2 + eps)
weight_sum = w1 + w2
w1 /= weight_sum
w2 /= weight_sum

valid_pred_ensemble = w1 * np.asarray(valid_pred_1) + w2 * np.asarray(valid_pred_2)
final_validation_score = mean_squared_error(valid_part[TARGET], valid_pred_ensemble) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

test_pred = w1 * np.asarray(test_pred_1) + w2 * np.asarray(test_pred_2)

submission = pd.DataFrame({"median_house_value": test_pred})
submission.to_csv("submission.csv", index=False)
