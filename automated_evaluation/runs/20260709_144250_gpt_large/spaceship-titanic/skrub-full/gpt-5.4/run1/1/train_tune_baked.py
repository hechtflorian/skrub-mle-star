
import os
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

lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.0246805592132731,
    num_leaves=26,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)

predictor_lgbm = (
    X_train
    .skb.apply(vectorizer_lgbm)
    .skb.apply(lgbm_model, y=y_train)
)

learner_lgbm = predictor_lgbm.skb.make_learner(fitted=True)

valid_pred_lgbm = pd.Series(
    learner_lgbm.predict({"data": valid_part}),
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
