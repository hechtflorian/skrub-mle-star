
import json
import math
import numpy as np
import pandas as pd
import skrub
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    open_date = pd.to_datetime(df["Open Date"], format="%m/%d/%Y")
    ref_date = pd.Timestamp("2015-01-01")
    age_days = (ref_date - open_date).dt.days
    df["restaurant_age_days"] = age_days
    df["restaurant_age_log"] = age_days.clip(lower=1).map(math.log)
    df = df.drop(columns=["Open Date"])
    return df


def main():
    train_path = "./input/train.csv"
    train_df = pd.read_csv(train_path)

    target_col = "revenue"
    random_state = 42
    n_iter = 5
    n_jobs = 1

    tune_plan = {
        "tunable_params": [
            {
                "name": "max_depth",
                "kind": "choose_int",
                "low": 3,
                "high": 5,
                "default": 4,
            },
            {
                "name": "learning_rate",
                "kind": "choose_float",
                "low": 0.02,
                "high": 0.05,
                "default": 0.03,
            },
        ]
    }

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=random_state
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    train_part = add_features(train_part)
    valid_part = add_features(valid_part)

    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    predictor = X_train.skb.apply(vectorizer).skb.apply(
        XGBRegressor(
            objective="reg:squarederror",
            n_estimators=250,
            max_depth=skrub.choose_int(3, 5, default=4, name="max_depth"),
            learning_rate=skrub.choose_float(
                0.02, 0.05, log=False, default=0.03, name="learning_rate"
            ),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbosity=0,
        ),
        y=y_train,
    )

    search = predictor.skb.make_randomized_search(
        n_iter=n_iter, n_jobs=n_jobs, random_state=random_state, fitted=True
    )
    search.fit({"data": train_part})

    valid_pred = search.best_learner_.predict({"data": valid_part})
    final_validation_score = mean_squared_error(
        valid_part[target_col], valid_pred
    ) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    raw_values = list(search.best_params_.values())
    remaining = list(raw_values)
    best_params = {}

    for spec in tune_plan["tunable_params"]:
        name = spec["name"]
        kind = spec["kind"]
        match = spec.get("default")

        if kind == "choose_int":
            for v in remaining:
                try:
                    candidate = int(round(float(v)))
                    if spec["low"] <= candidate <= spec["high"]:
                        match = candidate
                        remaining.remove(v)
                        break
                except Exception:
                    pass
            best_params[name] = int(match)

        elif kind == "choose_float":
            for v in remaining:
                try:
                    candidate = float(v)
                    if spec["low"] <= candidate <= spec["high"]:
                        match = candidate
                        remaining.remove(v)
                        break
                except Exception:
                    pass
            best_params[name] = float(match)

    print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))


if __name__ == "__main__":
    main()
