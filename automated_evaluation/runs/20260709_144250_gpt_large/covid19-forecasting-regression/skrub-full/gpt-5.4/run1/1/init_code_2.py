
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

INPUT_DIR = "./input"


def find_file(filename: str) -> str:
    for root, _, files in os.walk(INPUT_DIR):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"Could not find {filename} under {INPUT_DIR}")


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)
    return mean_squared_error(np.log1p(y_true), np.log1p(y_pred)) ** 0.5


def build_predictor(data_train, target_col):
    X = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )

    # Minimal fix: pass the skrub target DataOp directly instead of np.log1p(y)
    predictor = X.skb.apply(vectorizer).skb.apply(model, y=y)
    return predictor


def add_basic_features(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["day"] = out["Date"].dt.day
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["dayofyear"] = out["Date"].dt.dayofyear
    return out


def fit_predict_target(train_df, target_col, valid_part):
    data_train = skrub.var("data", train_df)
    predictor = build_predictor(data_train, target_col)
    learner = predictor.skb.make_learner(fitted=True)
    pred = learner.predict({"data": valid_part})
    pred = np.asarray(pred, dtype=float)
    pred = np.clip(pred, 0, None)
    return pred


def main():
    train_path = find_file("train.csv")
    test_path = find_file("test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = add_basic_features(train_df)
    test_df = add_basic_features(test_df)

    random_state = 42
    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=random_state
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    confirmed_pred = fit_predict_target(train_part, "ConfirmedCases", valid_part)
    fatalities_pred = fit_predict_target(train_part, "Fatalities", valid_part)

    score_confirmed = rmsle(valid_part["ConfirmedCases"], confirmed_pred)
    score_fatalities = rmsle(valid_part["Fatalities"], fatalities_pred)
    final_validation_score = (score_confirmed + score_fatalities) / 2.0

    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
