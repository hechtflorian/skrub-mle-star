
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


def robust_target_ratios(train_part):
    confirmed = np.asarray(train_part["ConfirmedCases"], dtype=float)
    fatal = np.asarray(train_part["Fatalities"], dtype=float)

    k_cf = np.median(confirmed / np.maximum(fatal, 1.0))
    k_fc = np.median(fatal / np.maximum(confirmed, 1.0))

    if not np.isfinite(k_cf):
        k_cf = 1.0
    if not np.isfinite(k_fc):
        k_fc = 0.0

    k_cf = max(k_cf, 0.0)
    k_fc = max(k_fc, 0.0)
    return k_cf, k_fc


def log_space_cross_blend(primary_pred, aux_pred, scale_ratio, weight):
    primary_pred = np.clip(np.asarray(primary_pred, dtype=float), 0, None)
    aux_scaled = np.clip(scale_ratio * np.asarray(aux_pred, dtype=float), 0, None)

    log_blend = (1.0 - weight) * np.log1p(primary_pred) + weight * np.log1p(aux_scaled)
    pred = np.expm1(log_blend)
    return np.clip(pred, 0, None)


def select_best_blend(y_true, primary_pred, aux_pred, scale_ratio, candidate_weights):
    best_pred = np.clip(np.asarray(primary_pred, dtype=float), 0, None)
    best_score = rmsle(y_true, best_pred)
    best_weight = 0.0

    for weight in candidate_weights:
        cand_pred = log_space_cross_blend(primary_pred, aux_pred, scale_ratio, weight)
        cand_score = rmsle(y_true, cand_pred)
        if cand_score < best_score:
            best_score = cand_score
            best_pred = cand_pred
            best_weight = weight

    return best_pred, best_score, best_weight


def main():
    train_path = "./input/train.csv"
    train_df = pd.read_csv(train_path)

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    pred_cases_base = train_and_predict_target(
        train_part, valid_part, "ConfirmedCases", random_state=42
    )
    pred_fatal_base = train_and_predict_target(
        train_part, valid_part, "Fatalities", random_state=43
    )

    k_cf, k_fc = robust_target_ratios(train_part)

    pred_cases_ens, score_cases, _ = select_best_blend(
        valid_part["ConfirmedCases"],
        pred_cases_base,
        pred_fatal_base,
        k_cf,
        candidate_weights=[0.00, 0.05, 0.10],
    )

    pred_fatal_ens, score_fatal, _ = select_best_blend(
        valid_part["Fatalities"],
        pred_fatal_base,
        pred_cases_base,
        k_fc,
        candidate_weights=[0.00, 0.10, 0.15, 0.20],
    )

    pred_cases_ens = np.clip(pred_cases_ens, 0, None)
    pred_fatal_ens = np.clip(pred_fatal_ens, 0, None)

    final_validation_score = (score_cases + score_fatal) / 2.0
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
