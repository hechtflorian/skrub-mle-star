
import os
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")

target_col = "Transported"
random_state = 42

tune_plan = {
    "tunable_params": [
        {
            "name": "learning_rate",
            "kind": "choose_float",
            "low": 0.02,
            "high": 0.05,
            "default": 0.03,
        },
        {
            "name": "num_leaves",
            "kind": "choose_int",
            "low": 24,
            "high": 40,
            "default": 31,
        },
    ]
}

train_df = pd.read_csv(TRAIN_PATH)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)


def engineer_passenger_features(df):
    out = df.copy()

    out = out.drop(columns=["PassengerId", "Name"], errors="ignore")

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype("string").str.split("/", n=2, expand=True)
        if cabin_parts is not None:
            if 0 in cabin_parts.columns:
                out["CabinDeck"] = cabin_parts[0]
            if 1 in cabin_parts.columns:
                out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
            if 2 in cabin_parts.columns:
                out["CabinSide"] = cabin_parts[2]
        out = out.drop(columns=["Cabin"], errors="ignore")

    return out


data_train_fe = data_train.skb.apply_func(engineer_passenger_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()

predictor_lgbm = (
    X_train
    .skb.apply(vectorizer_lgbm)
    .skb.apply(
        LGBMClassifier(
            n_estimators=250,
            learning_rate=skrub.choose_float(
                0.02, 0.05, default=0.03, name="learning_rate"
            ),
            num_leaves=skrub.choose_int(
                24, 40, default=31, name="num_leaves"
            ),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
        ),
        y=y_train,
    )
)

search = predictor_lgbm.skb.make_randomized_search(
    n_iter=5,
    n_jobs=1,
    random_state=random_state,
    fitted=True,
)
search.fit({"data": train_part})

valid_pred_lgbm = pd.Series(
    search.best_learner_.predict({"data": valid_part}),
    index=valid_part.index,
)

valid_true = pd.Series(valid_part[target_col], index=valid_part.index).astype(str).map(
    {"True": True, "False": False}
).fillna(valid_part[target_col])

valid_pred_lgbm_bool = valid_pred_lgbm.astype(str).map(
    {"True": True, "False": False}
).fillna(valid_pred_lgbm)

valid_pred_lgbm_num = pd.Series(
    valid_pred_lgbm_bool, index=valid_part.index
).astype(bool).astype(int)

ensemble_vote = valid_pred_lgbm_bool.astype(bool)

final_validation_score = accuracy_score(valid_true.astype(bool), ensemble_vote)
print(f"Final Validation Performance: {final_validation_score}")

raw_values = list(search.best_params_.values())
remaining = list(raw_values)
best_params = {}

for spec in tune_plan["tunable_params"]:
    name, kind = spec["name"], spec["kind"]

    if kind == "choose_int":
        match = next(
            (
                v
                for v in remaining
                if spec["low"] <= int(round(float(v))) <= spec["high"]
            ),
            spec.get("default"),
        )
        best_params[name] = int(round(float(match)))
    elif kind == "choose_float":
        match = next(
            (
                v
                for v in remaining
                if spec["low"] <= float(v) <= spec["high"]
            ),
            spec.get("default"),
        )
        best_params[name] = float(match)
    else:
        match = remaining[0] if remaining else spec.get("default")
        best_params[name] = match

    if match in remaining:
        remaining.remove(match)

print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
