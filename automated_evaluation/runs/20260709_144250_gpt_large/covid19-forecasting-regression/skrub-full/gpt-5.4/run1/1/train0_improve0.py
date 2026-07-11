
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

try:
    from lightgbm import LGBMRegressor
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    from lightgbm import LGBMRegressor


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)
    return mean_squared_log_error(y_true, y_pred) ** 0.5



def train_and_predict_target(train_part, valid_part, target_col, random_state=42):
    def engineer_features(df):
        out = df.copy()

        if "Id" in out.columns:
            out = out.drop(columns=["Id"])

        if "Date" in out.columns:
            date_parsed = pd.to_datetime(out["Date"], errors="coerce")
            out["DateOrdinal"] = date_parsed.map(
                lambda x: x.toordinal() if pd.notna(x) else np.nan
            ).astype(float)
            out["DateDay"] = date_parsed.dt.day.astype(float)
            out["DateWeek"] = date_parsed.dt.isocalendar().week.astype(float)
            out["DateMonth"] = date_parsed.dt.month.astype(float)
            out = out.drop(columns=["Date"])

        return out

    data_train = skrub.var("data", train_part)
    data_train_fe = data_train.skb.apply_func(engineer_features)

    X_train = (
        data_train_fe.drop(columns=["ConfirmedCases", "Fatalities"], errors="ignore")
        .skb.mark_as_X()
    )
    y_train = np.log1p(train_part[target_col]).astype(float)

    vectorizer = skrub.TableVectorizer()
    predictor = X_train.skb.apply(vectorizer).skb.apply(
        LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            verbose=-1,
        ),
        y=y_train,
    )

    learner = predictor.skb.make_learner(fitted=True)
    pred_log = learner.predict({"data": valid_part})
    pred = np.expm1(np.asarray(pred_log, dtype=float))
    pred = np.clip(pred, 0, None)
    return pred



def main():
    train_path = "./input/train.csv"
    test_path = "./input/test.csv"

    train_df = pd.read_csv(train_path)
    if os.path.exists(test_path):
        _ = pd.read_csv(test_path)

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    pred_cases = train_and_predict_target(train_part, valid_part, "ConfirmedCases", random_state=42)
    pred_fatal = train_and_predict_target(train_part, valid_part, "Fatalities", random_state=43)

    score_cases = rmsle(valid_part["ConfirmedCases"], pred_cases)
    score_fatal = rmsle(valid_part["Fatalities"], pred_fatal)
    final_validation_score = (score_cases + score_fatal) / 2.0

    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
