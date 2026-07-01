
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

train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=random_state)

train_data = skrub.var("data", train_part.reset_index(drop=True))
val_data = skrub.var("data", val_part.reset_index(drop=True))

X_train = train_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = train_data[target_col].skb.mark_as_y()

X_valid = val_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_valid = val_data[target_col].skb.mark_as_y()

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

baseline_pred = make_regression_graph(
    X_train,
    y_train,
    learning_rate=0.03,
    num_leaves=31,
)

tuned_pred = make_regression_graph(
    X_train,
    y_train,
    learning_rate=learning_rate_choice,
    num_leaves=num_leaves_choice,
)

baseline_learner = baseline_pred.skb.make_learner(fitted=True)
baseline_val_predictions = baseline_learner.predict({"data": val_part.drop(columns=[target_col], errors="ignore")})

tuned_learner = tuned_pred.skb.make_learner(fitted=True)
tuned_val_predictions = tuned_learner.predict({"data": val_part.drop(columns=[target_col], errors="ignore")})

baseline_val_score = float(mean_squared_error(val_part[target_col], baseline_val_predictions) ** 0.5)
tuned_val_score = float(mean_squared_error(val_part[target_col], tuned_val_predictions) ** 0.5)

def rank_normalize(train_ref, pred_ref, pred_eval):
    train_ref = np.asarray(train_ref).ravel()
    pred_ref = np.asarray(pred_ref).ravel()
    pred_eval = np.asarray(pred_eval).ravel()
    order = np.argsort(pred_ref)
    sorted_train = np.sort(train_ref)
    if len(sorted_train) == 1:
        return np.full_like(pred_eval, fill_value=float(sorted_train[0]), dtype=float)
    quantiles = np.linspace(0.0, 1.0, len(sorted_train))
    pred_eval_rank = pd.Series(pred_eval).rank(method="average").to_numpy()
    pred_eval_q = (pred_eval_rank - 1.0) / max(len(pred_eval) - 1.0, 1.0)
    pred_eval_q = np.clip(pred_eval_q, 0.0, 1.0)
    return np.interp(pred_eval_q, quantiles, sorted_train)

def blend_predictions(anchor_val, other_val, anchor_test, other_test, y_true):
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    best_alpha = 0.0
    best_score = float("inf")

    anchor_val = np.asarray(anchor_val).ravel()
    other_val = np.asarray(other_val).ravel()
    anchor_test = np.asarray(anchor_test).ravel()
    other_test = np.asarray(other_test).ravel()
    y_true = np.asarray(y_true).ravel()

    for alpha in alphas:
        raw_val = anchor_val + alpha * (other_val - anchor_val)
        raw_score = float(mean_squared_error(y_true, raw_val) ** 0.5)

        anchor_rank_val = rank_normalize(y_true, anchor_val, anchor_val)
        other_rank_val = rank_normalize(y_true, other_val, other_val)
        anchor_rank_test = rank_normalize(y_true, anchor_val, anchor_test)
        other_rank_test = rank_normalize(y_true, other_val, other_test)

        rank_blend_val = anchor_rank_val + alpha * (other_rank_val - anchor_rank_val)
        rank_blend_score = float(mean_squared_error(y_true, rank_blend_val) ** 0.5)

        blended_val = 0.5 * raw_val + 0.5 * rank_blend_val
        score = float(mean_squared_error(y_true, blended_val) ** 0.5)

        if score < best_score:
            best_score = score
            best_alpha = alpha

    raw_test = anchor_test + best_alpha * (other_test - anchor_test)

    anchor_rank_val = rank_normalize(y_true, anchor_val, anchor_val)
    other_rank_val = rank_normalize(y_true, other_val, other_val)
    anchor_rank_test = rank_normalize(y_true, anchor_val, anchor_test)
    other_rank_test = rank_normalize(y_true, other_val, other_test)

    rank_blend_test = anchor_rank_test + best_alpha * (other_rank_test - anchor_rank_test)
    final_test = 0.5 * raw_test + 0.5 * rank_blend_test

    return best_alpha, best_score, final_test

if tuned_val_score <= baseline_val_score:
    anchor_val_predictions = tuned_val_predictions
    other_val_predictions = baseline_val_predictions
    anchor_test_learner = tuned_learner
    other_test_learner = baseline_learner
    anchor_name = "tuned"
else:
    anchor_val_predictions = baseline_val_predictions
    other_val_predictions = tuned_val_predictions
    anchor_test_learner = baseline_learner
    other_test_learner = tuned_learner
    anchor_name = "baseline"

anchor_test_predictions = anchor_test_learner.predict({"data": test_df})
other_test_predictions = other_test_learner.predict({"data": test_df})

best_alpha, final_validation_score, test_predictions = blend_predictions(
    anchor_val_predictions,
    other_val_predictions,
    anchor_test_predictions,
    other_test_predictions,
    val_part[target_col],
)

train_target_low = float(train_df[target_col].quantile(0.01))
train_target_high = float(train_df[target_col].quantile(0.99))
test_predictions = np.clip(np.asarray(test_predictions).ravel(), train_target_low, train_target_high)

print(f"Final Validation Performance: {final_validation_score}")

full_data = skrub.var("data", train_df)
X_full = full_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = full_data[target_col].skb.mark_as_y()

final_pred = make_regression_graph(
    X_full,
    y_full,
    learning_rate=0.03,
    num_leaves=31,
)

final_learner = final_pred.skb.make_learner(fitted=True)
_ = final_learner.predict({"data": test_df})

submission = pd.DataFrame({"median_house_value": test_predictions})
submission.to_csv("submission.csv", index=False)
print(submission.head())
