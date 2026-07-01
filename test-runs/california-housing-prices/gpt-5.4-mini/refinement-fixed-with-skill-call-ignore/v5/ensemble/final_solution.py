
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
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

DATA_DIR = "./input"
FINAL_DIR = "./final"
os.makedirs(FINAL_DIR, exist_ok=True)

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
baseline_test_predictions = baseline_learner.predict({"data": test_df})

tuned_learner = tuned_pred.skb.make_learner(fitted=True)
tuned_test_predictions = tuned_learner.predict({"data": test_df})

train_target_low = float(train_df[target_col].quantile(0.01))
train_target_high = float(train_df[target_col].quantile(0.99))

final_test_predictions = 0.5 * np.asarray(baseline_test_predictions).ravel() + 0.5 * np.asarray(tuned_test_predictions).ravel()
final_test_predictions = np.clip(final_test_predictions, train_target_low, train_target_high)

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

submission = pd.DataFrame({"median_house_value": final_test_predictions})
submission.to_csv(os.path.join(FINAL_DIR, "submission.csv"), index=False)
print(submission.head())
