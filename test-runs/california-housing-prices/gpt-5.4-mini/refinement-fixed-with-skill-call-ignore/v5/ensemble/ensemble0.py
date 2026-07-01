
import os
import sys
import subprocess
import warnings

warnings.filterwarnings("ignore")

def ensure_package(pkg_name, import_name=None):
    mod = import_name or pkg_name
    try:
        __import__(mod)
        return
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_name])

ensure_package("lightgbm")
ensure_package("skrub")

import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

DATA_DIR = "./input"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "median_house_value"
random_state = 42
epsilon = 1e-8

# Shared split for both models so validation RMSE is comparable.
train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=random_state)
train_part = train_part.reset_index(drop=True)
val_part = val_part.reset_index(drop=True)

# Build a single reusable DataOps graph shape.
def make_regression_graph(X_data, y_data, learning_rate, num_leaves):
    vectorizer = skrub.TableVectorizer()
    regressor = LGBMRegressor(
        n_estimators=700,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        n_jobs=-1,
    )
    return X_data.skb.apply(vectorizer).skb.apply(regressor, y=y_data)

def fit_and_predict_variant(train_df_local, valid_df_local, test_df_local, learning_rate, num_leaves):
    data = skrub.var("data", train_df_local)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    pred_graph = make_regression_graph(X, y, learning_rate=learning_rate, num_leaves=num_leaves)
    learner = pred_graph.skb.make_learner(fitted=True)

    valid_features = valid_df_local.drop(columns=[target_col], errors="ignore")
    val_pred = learner.predict({"data": valid_features})
    test_pred = learner.predict({"data": test_df_local})

    return np.asarray(val_pred), np.asarray(test_pred), pred_graph

# Baseline path retained with strong regularization.
baseline_val_pred, baseline_test_pred, baseline_graph = fit_and_predict_variant(
    train_part,
    val_part,
    test_df,
    learning_rate=0.03,
    num_leaves=31,
)

baseline_val_score = float(mean_squared_error(val_part[target_col], baseline_val_pred) ** 0.5)

# Tuned path stays DataOps-native via in-graph choices.
learning_rate_choice = skrub.choose_from(
    {
        "lr_0p02": 0.02,
        "lr_0p03": 0.03,
        "lr_0p05": 0.05,
    },
    name="learning_rate",
)

num_leaves_choice = skrub.choose_from(
    {
        "leaves_31": 31,
        "leaves_47": 47,
        "leaves_63": 63,
    },
    name="num_leaves",
)

tuned_val_pred, tuned_test_pred, tuned_graph = fit_and_predict_variant(
    train_part,
    val_part,
    test_df,
    learning_rate=learning_rate_choice,
    num_leaves=num_leaves_choice,
)

tuned_val_score = float(mean_squared_error(val_part[target_col], tuned_val_pred) ** 0.5)

# Inverse-RMSE weighting.
w_baseline = 1.0 / (baseline_val_score + epsilon)
w_tuned = 1.0 / (tuned_val_score + epsilon)

val_blend = (w_baseline * baseline_val_pred + w_tuned * tuned_val_pred) / (w_baseline + w_tuned)
test_blend = (w_baseline * baseline_test_pred + w_tuned * tuned_test_pred) / (w_baseline + w_tuned)

final_validation_score = float(mean_squared_error(val_part[target_col], val_blend) ** 0.5)

# Clip to a reasonable range based on training target quantiles.
low_clip = float(train_df[target_col].quantile(0.01))
high_clip = float(train_df[target_col].quantile(0.99))
test_blend = np.clip(test_blend, low_clip, high_clip)

print(f"Baseline Validation Performance: {baseline_val_score}")
print(f"Tuned Validation Performance: {tuned_val_score}")
print(f"Final Validation Performance: {final_validation_score}")

submission = pd.DataFrame({"median_house_value": test_blend})
submission.to_csv("submission.csv", index=False)
print(submission.head())
