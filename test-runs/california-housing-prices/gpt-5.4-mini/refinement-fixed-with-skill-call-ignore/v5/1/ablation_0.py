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
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor

DATA_DIR = "./input"
train_path = os.path.join(DATA_DIR, "train.csv")

train_df = pd.read_csv(train_path)
target_col = "median_house_value"

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].reset_index(drop=True)
val_part = train_df.iloc[val_idx].reset_index(drop=True)

def rmse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred) ** 0.5)

def fit_eval(train_frame, val_frame, vectorizer, model_params):
    train_data = skrub.var("data", train_frame)
    X_train = train_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = train_data[target_col].skb.mark_as_y()

    valid_data = skrub.var("data", val_frame)
    X_valid = valid_data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_valid = valid_data[target_col].skb.mark_as_y()

    pred = X_train.skb.apply(vectorizer).skb.apply(
        LGBMRegressor(**model_params),
        y=y_train,
    )
    learner = pred.skb.make_learner(fitted=True)
    val_pred = learner.predict({"data": val_frame.drop(columns=[target_col], errors="ignore")})
    return rmse(val_frame[target_col], val_pred)

baseline_params = dict(
    n_estimators=700,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)

ablation_results = {}

# Baseline: full original-ish setup
ablation_results["baseline"] = fit_eval(
    train_part,
    val_part,
    skrub.TableVectorizer(),
    baseline_params,
)

# Ablation 1: disable preview-style regularization by changing subsample/colsample to 1.0
no_sampling_params = dict(baseline_params)
no_sampling_params["subsample"] = 1.0
no_sampling_params["colsample_bytree"] = 1.0
ablation_results["no_sampling_regularization"] = fit_eval(
    train_part,
    val_part,
    skrub.TableVectorizer(),
    no_sampling_params,
)

# Ablation 2: reduce model complexity by using fewer trees
smaller_model_params = dict(baseline_params)
smaller_model_params["n_estimators"] = 300
ablation_results["fewer_trees"] = fit_eval(
    train_part,
    val_part,
    skrub.TableVectorizer(),
    smaller_model_params,
)

for name, score in ablation_results.items():
    print(f"Ablation[{name}] RMSE: {score}")

best_variant = min(ablation_results, key=ablation_results.get)
best_score = ablation_results[best_variant]

baseline_score = ablation_results["baseline"]
print(f"Final Validation Performance: {baseline_score}")
print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")

deltas = {
    name: score - baseline_score
    for name, score in ablation_results.items()
    if name != "baseline"
}
most_contributive = min(deltas, key=lambda k: deltas[k]) if deltas else "baseline"
if deltas:
    print(
        f"Most performance-contributing change: {most_contributive} "
        f"(RMSE change vs baseline: {deltas[most_contributive]:+.6f})"
    )
else:
    print("Most performance-contributing change: baseline only")