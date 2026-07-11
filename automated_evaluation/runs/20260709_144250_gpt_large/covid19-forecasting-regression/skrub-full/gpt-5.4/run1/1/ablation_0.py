
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_DIR = "./input"


def find_file(filename):
    candidates = [
        os.path.join(DATA_DIR, filename),
        os.path.join(DATA_DIR, "covid19-global-forecasting-week-1", filename),
        os.path.join(DATA_DIR, "covid19-global-forecasting-week-2", filename),
        os.path.join(DATA_DIR, "covid19-global-forecasting-week-3", filename),
        os.path.join(DATA_DIR, "covid19-global-forecasting-week-4", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(DATA_DIR, filename)


train_df = pd.read_csv(find_file("train.csv"))

train_df["Province_State"] = train_df["Province_State"].fillna("")
train_df["Country_Region"] = train_df["Country_Region"].fillna("")
train_df["Date"] = pd.to_datetime(train_df["Date"])
train_df["DateOrdinal"] = train_df["Date"].map(pd.Timestamp.toordinal)
train_df["Location"] = train_df["Country_Region"] + "_" + train_df["Province_State"]

random_state = 42
test_size = 0.2
metric_label = "RMSLE"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def build_graph(data_train, target_col, use_location=True):
    feature_cols = ["Province_State", "Country_Region", "DateOrdinal"]
    if use_location:
        feature_cols.append("Location")
    X = data_train[feature_cols].skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state,
        verbose=-1,
    )
    return X.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y)


def score_variant(variant_name, target_col, use_location):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train, target_col, use_location=use_location)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = rmsle(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {target_col}_{metric_label}: {score}")
    return score


scores = {}
variant_defs = [
    ("baseline", True),
    ("no_location", False),
]

for variant_name, use_location in variant_defs:
    cc_score = score_variant(variant_name, "ConfirmedCases", use_location)
    ft_score = score_variant(variant_name, "Fatalities", use_location)
    combined_score = (cc_score + ft_score) / 2.0
    scores[variant_name] = combined_score
    print(f"Ablation[{variant_name}] Combined_{metric_label}: {combined_score}")

best_variant = min(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | Combined_{metric_label}: {best_score}")

final_validation_score = best_score
print(f"Final Validation Performance: {final_validation_score}")
