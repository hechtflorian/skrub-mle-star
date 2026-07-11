
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor

random_state = 42
target_col = "count"

train_df = pd.read_csv("./input/train.csv")
train_df["datetime"] = pd.to_datetime(train_df["datetime"])
train_df = train_df.sort_values("datetime").reset_index(drop=True)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def build_learner(train_part, seed_offset=0):
    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=random_state + seed_offset,
        n_jobs=-1,
        verbose=-1,
    )

    pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    return pred.skb.make_learner(fitted=True)


learner_1 = build_learner(train_part, seed_offset=0)
learner_2 = build_learner(train_part, seed_offset=1)
learner_3 = build_learner(train_part, seed_offset=2)

valid_pred_1 = np.maximum(learner_1.predict({"data": valid_part}), 0)
valid_pred_2 = np.maximum(learner_2.predict({"data": valid_part}), 0)
valid_pred_3 = np.maximum(learner_3.predict({"data": valid_part}), 0)

valid_blend = 0.5 * valid_pred_1 + 0.3 * valid_pred_2 + 0.2 * valid_pred_3
valid_blend = np.maximum(valid_blend, 0)

final_validation_score = mean_squared_log_error(valid_part[target_col], valid_blend) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
