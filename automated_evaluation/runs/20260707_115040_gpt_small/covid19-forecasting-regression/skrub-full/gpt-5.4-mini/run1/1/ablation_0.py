import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor

random_state = 42
test_size = 0.2

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

target_cols = ["ConfirmedCases", "Fatalities"]


def rmsle(y_true, y_pred):
    y_true = np.maximum(np.asarray(y_true), 0)
    y_pred = np.maximum(np.asarray(y_pred), 0)
    return mean_squared_log_error(y_true, y_pred) ** 0.5


train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_date_features(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["day"] = out["Date"].dt.day
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["dayofyear"] = out["Date"].dt.dayofyear
    out["weekofyear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["date_ordinal"] = out["Date"].map(pd.Timestamp.toordinal)
    out["region"] = out["Country_Region"].fillna("") + "_" + out["Province_State"].fillna("")
    out = out.drop(columns=["Date"])
    return out


def add_date_features_no_region(df):
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out["year"] = out["Date"].dt.year
    out["month"] = out["Date"].dt.month
    out["day"] = out["Date"].dt.day
    out["dayofweek"] = out["Date"].dt.dayofweek
    out["dayofyear"] = out["Date"].dt.dayofyear
    out["weekofyear"] = out["Date"].dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out["date_ordinal"] = out["Date"].map(pd.Timestamp.toordinal)
    out = out.drop(columns=["Date"])
    return out


def score_variant(variant_name, fe_func=None, drop_cols=None, vectorizer_kwargs=None):
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_cols, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    return None


def run_variant(variant_name, fe_func, drop_region=False, vectorizer_mode="default"):
    data_train = skrub.var("data", train_part)
    data_train = data_train.skb.apply_func(fe_func)

    X_train = data_train.drop(columns=target_cols, errors="ignore").skb.mark_as_X()
    y_train = data_train["ConfirmedCases"].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    if vectorizer_mode == "drop_high":
        vectorizer = skrub.TableVectorizer(high_cardinality="drop")
    elif vectorizer_mode == "drop_low":
        vectorizer = skrub.TableVectorizer(low_cardinality="drop")

    if drop_region:
        X_train = X_train.skb.apply(skrub.DropCols(cols=["region"]))

    model = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=63,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )

    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = rmsle(valid_part["ConfirmedCases"].values, valid_pred)
    print(f"Ablation[{variant_name}] RMSLE: {score}")
    return score


scores = {}

# Baseline: original feature engineering
scores["baseline"] = run_variant("baseline", add_date_features)

# Ablation 1: remove the engineered region feature
scores["no_region_feature"] = run_variant("no_region_feature", add_date_features_no_region)

# Ablation 2: drop the high-cardinality region-like string features at vectorization time
scores["drop_high_cardinality"] = run_variant("drop_high_cardinality", add_date_features, vectorizer_mode="drop_high")

best_variant = min(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | RMSLE: {best_score}")