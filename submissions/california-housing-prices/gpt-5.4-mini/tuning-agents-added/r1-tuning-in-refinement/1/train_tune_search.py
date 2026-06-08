
import json
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


@skrub.deferred
def add_ratio_features(df):
    out = df.copy()

    households = out["households"].replace(0, np.nan) if "households" in out.columns else np.nan
    population = out["population"].replace(0, np.nan) if "population" in out.columns else np.nan

    if "total_rooms" in out.columns and "households" in out.columns:
        out["rooms_per_household"] = (out["total_rooms"] / households).replace([np.inf, -np.inf], np.nan)
    if "total_bedrooms" in out.columns and "households" in out.columns:
        out["bedrooms_per_household"] = (out["total_bedrooms"] / households).replace([np.inf, -np.inf], np.nan)
    if "population" in out.columns and "households" in out.columns:
        out["population_per_household"] = (out["population"] / households).replace([np.inf, -np.inf], np.nan)
    if "total_rooms" in out.columns and "population" in out.columns:
        out["rooms_per_person"] = (out["total_rooms"] / population).replace([np.inf, -np.inf], np.nan)
    if "total_bedrooms" in out.columns and "total_rooms" in out.columns:
        out["bedroom_room_ratio"] = (
            out["total_bedrooms"] / out["total_rooms"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

    return out


def build_pred(train_frame):
    data = skrub.var("data", train_frame)
    data_fe = data.skb.apply_func(add_ratio_features)

    X = data_fe.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data_fe[TARGET_COL].skb.mark_as_y()

    X_clean = X.skb.apply(skrub.Cleaner(drop_if_constant=True))
    X_num = X_clean.skb.select(skrub.selectors.numeric())
    X_num_imputed = X_num.skb.apply(SimpleImputer(strategy="median"))

    iterations = skrub.choose_int(1800, 3200, name="iterations")
    learning_rate = skrub.choose_float(0.015, 0.05, name="learning_rate")
    depth = skrub.choose_int(6, 9, name="depth")
    l2_leaf_reg = skrub.choose_float(2.0, 12.0, name="l2_leaf_reg")

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=0,
        subsample=0.85,
        bootstrap_type="Bernoulli",
        l2_leaf_reg=l2_leaf_reg,
    )

    pred = X_num_imputed.skb.apply(model, y=y)
    return pred


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE)

    pred = build_pred(tr_df)
    search = pred.skb.make_randomized_search(n_iter=5, n_jobs=2, random_state=42, fitted=True)
    best_learner = search.best_learner_

    val_pred = best_learner.predict({"data": va_df})
    final_validation_score = rmse(va_df[TARGET_COL].values, val_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    best_params = getattr(search, "best_params_", None)
    if best_params is None:
        best_params = getattr(search, "best_params", {})
    print(f"TUNING_BEST_PARAMS: {json.dumps(best_params)}")

    final_pred = build_pred(train_df)
    final_search = final_pred.skb.make_randomized_search(n_iter=5, n_jobs=2, random_state=42, fitted=True)
    final_learner = final_search.best_learner_
    test_predictions = final_learner.predict({"data": test_df})

    submission = pd.DataFrame({TARGET_COL: test_predictions})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
