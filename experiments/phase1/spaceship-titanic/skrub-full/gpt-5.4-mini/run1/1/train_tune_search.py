
import sys
import subprocess
import warnings
import json

subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "lightgbm", "-q"])

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_variants = {
    "cb_d4_lr0.03_it100": dict(depth=4, learning_rate=0.03, iterations=100),
    "cb_d4_lr0.05_it100": dict(depth=4, learning_rate=0.05, iterations=100),
    "cb_d5_lr0.03_it100": dict(depth=5, learning_rate=0.03, iterations=100),
    "cb_d5_lr0.05_it100": dict(depth=5, learning_rate=0.05, iterations=100),
}
lgb_variants = {
    "lgb_le150_lr0.03": dict(n_estimators=150, learning_rate=0.03),
    "lgb_le150_lr0.05": dict(n_estimators=150, learning_rate=0.05),
    "lgb_le250_lr0.03": dict(n_estimators=250, learning_rate=0.03),
    "lgb_le250_lr0.05": dict(n_estimators=250, learning_rate=0.05),
}

cat_model = skrub.choose_from(
    {
        k: CatBoostClassifier(
            verbose=0,
            loss_function="Logloss",
            random_seed=42,
            n_jobs=1,
            **v,
        )
        for k, v in cat_variants.items()
    },
    name="cat_model_variant",
)

lgb_model = skrub.choose_from(
    {
        k: LGBMClassifier(
            random_state=42,
            verbose=-1,
            n_jobs=1,
            **v,
        )
        for k, v in lgb_variants.items()
    },
    name="lgb_model_variant",
)

cat_pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
lgb_pred = X_train.skb.apply(vectorizer).skb.apply(lgb_model, y=y_train)

pred = cat_pred + lgb_pred

search = pred.skb.make_randomized_search(n_iter=4, n_jobs=1, random_state=42, fitted=True)
search.fit({"data": train_part})

cat_valid_pred = np.asarray(search.best_learner_.predict({"data": valid_part})).ravel()
lgb_valid_pred = np.asarray(search.best_learner_.predict({"data": valid_part})).ravel()

valid_pred = cat_valid_pred >= 0.5 if cat_valid_pred.dtype != bool else cat_valid_pred
valid_pred = valid_pred.astype(float) if valid_pred.dtype == bool else valid_pred
ensemble_valid_pred = valid_pred >= 0.5

final_validation_score = accuracy_score(valid_part[target_col], ensemble_valid_pred)
print(f"Final Validation Performance: {final_validation_score}")

best_params = {
    "cat_model_variant": next(
        k for k, v in cat_variants.items() if v == search.best_params_.get("cat_model_variant", search.best_params_.get("data_op__0", {}))
    ) if isinstance(search.best_params_, dict) else "cb_d5_lr0.03_it100",
    "lgb_model_variant": next(
        k for k, v in lgb_variants.items() if v == search.best_params_.get("lgb_model_variant", search.best_params_.get("data_op__1", {}))
    ) if isinstance(search.best_params_, dict) else "lgb_le250_lr0.05",
}
if isinstance(search.best_params_, dict):
    for spec_name, variants in [("cat_model_variant", cat_variants), ("lgb_model_variant", lgb_variants)]:
        for k, v in variants.items():
            if all(str(v.get(pk)) == str(search.best_params_.get(spec_name, search.best_params_.get(f"data_op__{0 if spec_name=='cat_model_variant' else 1}", {})).get(pk)) for pk in v.keys()):
                best_params[spec_name] = k
                break

print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
