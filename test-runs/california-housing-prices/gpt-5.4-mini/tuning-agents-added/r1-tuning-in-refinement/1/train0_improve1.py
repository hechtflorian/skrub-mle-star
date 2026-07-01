

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
import skrub.selectors as s
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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

    denom = out.get("households", pd.Series(index=out.index, dtype="float64")).replace(0, np.nan)
    out["rooms_per_household"] = out.get("total_rooms", 0) / denom
    out["bedrooms_per_room"] = out.get("total_bedrooms", 0) / out.get("total_rooms", np.nan).replace(0, np.nan)
    out["population_per_household"] = out.get("population", 0) / denom
    out["population_per_room"] = out.get("population", 0) / out.get("total_rooms", np.nan).replace(0, np.nan)
    out["bedrooms_per_household"] = out.get("total_bedrooms", 0) / denom
    out["rooms_per_person"] = out.get("total_rooms", 0) / out.get("population", np.nan).replace(0, np.nan)

    for col in [
        "rooms_per_household",
        "bedrooms_per_room",
        "population_per_household",
        "population_per_room",
        "bedrooms_per_household",
        "rooms_per_person",
    ]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


def build_feature_graph(data_var):
    data_fe = data_var.skb.apply_func(add_ratio_features)
    X = data_fe.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data_fe[TARGET_COL].skb.mark_as_y()

    X = X.skb.apply(skrub.Cleaner(drop_if_constant=True))
    X = X.skb.apply(skrub.ApplyToCols(skrub.SquashingScaler(max_absolute_value=3), cols=s.numeric()))
    X = X.skb.apply(skrub.ApplyToCols(SimpleImputer(strategy="median"), cols=s.numeric(), allow_reject=True))

    vectorizer = skrub.TableVectorizer()
    X_vec = X.skb.apply(vectorizer)
    return X_vec, y


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE)

    data = skrub.var("data", tr_df)
    X_vec, y = build_feature_graph(data)

    model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=10.0,
        min_data_in_leaf=20,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )

    learner = X_vec.skb.apply(model, y=y).skb.make_learner(fitted=True)
    val_pred = learner.predict({"data": va_df})
    final_validation_score = rmse(va_df[TARGET_COL].values, val_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    full_data = skrub.var("full_data", train_df)
    full_X_vec, full_y = build_feature_graph(full_data)

    final_model = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=10.0,
        min_data_in_leaf=20,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=0,
        allow_writing_files=False,
    )

    final_learner = full_X_vec.skb.apply(final_model, y=full_y).skb.make_learner(fitted=True)
    test_predictions = final_learner.predict({"full_data": test_df})

    submission = pd.DataFrame({TARGET_COL: test_predictions})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
