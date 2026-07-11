
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

RANDOM_STATE = 42
TARGET_COL = "Attrition"
ID_COL = "id"

tune_plan = {
    "tunable_params": [
        {
            "name": "lgbm_num_leaves",
            "kind": "choose_int",
            "low": 15,
            "high": 63,
            "default": 31,
        },
        {
            "name": "lgbm_learning_rate",
            "kind": "choose_float",
            "low": 0.03,
            "high": 0.12,
            "default": 0.1,
            "log": True,
        },
    ]
}

train_path = "./input/train.csv"
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=train_df[TARGET_COL],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
y_train = data_train[TARGET_COL].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

cat_X_train = X_train.skb.apply(vectorizer)
cat_model = CatBoostClassifier(
    random_state=RANDOM_STATE,
    verbose=0,
    cat_features=None,
)
cat_predictor = cat_X_train.skb.apply(cat_model, y=y_train)

lgbm_X_train = X_train.skb.apply(vectorizer)
lgbm_predictor = lgbm_X_train.skb.apply(
    LGBMClassifier(
        random_state=RANDOM_STATE,
        verbose=-1,
        n_estimators=50,
        num_leaves=skrub.choose_int(
            15, 63, default=31, name="lgbm_num_leaves"
        ),
        learning_rate=skrub.choose_float(
            0.03, 0.12, default=0.1, log=True, name="lgbm_learning_rate"
        ),
    ),
    y=y_train,
)

search = lgbm_predictor.skb.make_randomized_search(
    n_iter=5,
    n_jobs=1,
    random_state=RANDOM_STATE,
    fitted=True,
)
search.fit({"data": train_part})

cat_learner = cat_predictor.skb.make_learner(fitted=True)

lgbm_valid_pred = np.asarray(search.best_learner_.predict({"data": valid_part}))
cat_valid_pred = np.asarray(cat_learner.predict({"data": valid_part}))

if set(np.unique(lgbm_valid_pred)).issubset({0, 1}):
    lgbm_valid_pred = search.best_learner_.predict_proba({"data": valid_part})[:, 1]
if set(np.unique(cat_valid_pred)).issubset({0, 1}):
    cat_valid_pred = cat_learner.predict_proba({"data": valid_part})[:, 1]

ensemble_valid_pred = 0.5 * np.asarray(cat_valid_pred) + 0.5 * np.asarray(lgbm_valid_pred)
final_validation_score = roc_auc_score(valid_part[TARGET_COL], ensemble_valid_pred)

raw_values = list(search.best_params_.values())
remaining = list(raw_values)
best_params = {}

for spec in tune_plan["tunable_params"]:
    name = spec["name"]
    kind = spec["kind"]
    match = None

    if kind == "choose_int":
        for v in remaining:
            try:
                iv = int(round(float(v)))
                if spec["low"] <= iv <= spec["high"]:
                    match = iv
                    break
            except Exception:
                pass
        if match is None:
            match = int(spec.get("default"))
        best_params[name] = int(match)
        for v in list(remaining):
            try:
                if int(round(float(v))) == int(match):
                    remaining.remove(v)
                    break
            except Exception:
                pass

    elif kind == "choose_float":
        for v in remaining:
            try:
                fv = float(v)
                if spec["low"] <= fv <= spec["high"]:
                    match = fv
                    break
            except Exception:
                pass
        if match is None:
            match = float(spec.get("default"))
        best_params[name] = float(match)
        for v in list(remaining):
            try:
                if float(v) == float(match):
                    remaining.remove(v)
                    break
            except Exception:
                pass

print(f"Final Validation Performance: {final_validation_score}")
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
