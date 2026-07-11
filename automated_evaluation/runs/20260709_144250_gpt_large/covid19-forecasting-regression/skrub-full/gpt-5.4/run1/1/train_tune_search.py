
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import skrub
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "skrub"])
    import skrub

try:
    from lightgbm import LGBMRegressor
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split


def rmsle(y_true, y_pred):
    y_true = np.clip(np.asarray(y_true, dtype=float), 0, None)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def fe_func(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["Year"] = out["Date"].dt.year
    out["Month"] = out["Date"].dt.month
    out["Day"] = out["Date"].dt.day
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["DayOfYear"] = out["Date"].dt.dayofyear

    out["Province_State"] = out["Province_State"].fillna("")
    out["Country_Region"] = out["Country_Region"].fillna("")

    out["Location"] = out["Country_Region"].astype(str) + "__" + out["Province_State"].astype(str)
    return out


def build_search(train_part, target_col, random_state=42):
    data_train = skrub.var("data", train_part)
    data_train_fe = data_train.skb.apply_func(fe_func)

    X_train = data_train_fe.drop(columns=["ConfirmedCases", "Fatalities", "Id"], errors="ignore").skb.mark_as_X()
    y_train = data_train_fe[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    model = LGBMRegressor(
        n_estimators=skrub.choose_int(100, 300, n_steps=3, name=f"{target_col}_n_estimators"),
        learning_rate=skrub.choose_float(0.03, 0.15, log=True, default=0.05, name=f"{target_col}_learning_rate"),
        num_leaves=skrub.choose_int(15, 63, n_steps=4, name=f"{target_col}_num_leaves"),
        subsample=skrub.choose_float(0.7, 1.0, default=0.9, name=f"{target_col}_subsample"),
        colsample_bytree=skrub.choose_float(0.7, 1.0, default=0.9, name=f"{target_col}_colsample"),
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )

    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    search = pred.skb.make_randomized_search(
        n_iter=4, n_jobs=1, random_state=random_state, fitted=True
    )
    return search


def build_fixed_learner(train_part, target_col, best_params, random_state=42):
    data_train = skrub.var("data", train_part)
    data_train_fe = data_train.skb.apply_func(fe_func)

    X_train = data_train_fe.drop(columns=["ConfirmedCases", "Fatalities", "Id"], errors="ignore").skb.mark_as_X()
    y_train = data_train_fe[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()

    predictor = X_train.skb.apply(vectorizer).skb.apply(
        LGBMRegressor(
            n_estimators=int(best_params.get(f"{target_col}_n_estimators", 200)),
            learning_rate=float(best_params.get(f"{target_col}_learning_rate", 0.05)),
            num_leaves=int(best_params.get(f"{target_col}_num_leaves", 31)),
            subsample=float(best_params.get(f"{target_col}_subsample", 0.9)),
            colsample_bytree=float(best_params.get(f"{target_col}_colsample", 0.9)),
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        ),
        y=y_train,
    )
    return predictor.skb.make_learner(fitted=True)


def extract_best_params(search, target_col):
    best = {}
    for k, v in search.best_params_.items():
        key = str(k)
        if target_col not in key:
            continue
        if isinstance(v, (np.integer,)):
            best[key] = int(v)
        elif isinstance(v, (np.floating,)):
            best[key] = float(v)
        else:
            best[key] = v
    return best


def main():
    train_df = pd.read_csv("./input/train.csv")

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=42
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    search = build_search(train_part, "ConfirmedCases", random_state=42)
    search.fit({"data": train_part})

    search_fatal = build_search(train_part, "Fatalities", random_state=43)
    search_fatal.fit({"data": train_part})

    best_params_cases = extract_best_params(search, "ConfirmedCases")
    best_params_fatal = extract_best_params(search_fatal, "Fatalities")

    learner_cases = build_fixed_learner(train_part, "ConfirmedCases", best_params_cases, random_state=42)
    learner_fatal = build_fixed_learner(train_part, "Fatalities", best_params_fatal, random_state=43)

    pred_cases = learner_cases.predict({"data": valid_part})
    pred_fatal = learner_fatal.predict({"data": valid_part})

    score_cases = rmsle(valid_part["ConfirmedCases"], pred_cases)
    score_fatal = rmsle(valid_part["Fatalities"], pred_fatal)
    final_validation_score = (score_cases + score_fatal) / 2.0

    print(f"Final Validation Performance: {final_validation_score}")
    print("TUNING_BEST_PARAMS:", json.dumps({
        "ConfirmedCases": best_params_cases,
        "Fatalities": best_params_fatal,
    }, default=str))


if __name__ == "__main__":
    main()
