
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "booking_status"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def train_one_leg(leg_df):
    data_train = skrub.var("data", leg_df)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = CatBoostClassifier(
        random_state=random_state,
        verbose=0,
    )

    predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    learner = predictor.skb.make_learner(fitted=True)
    return learner


def predict_positive_proba(learner, df):
    pred = learner.predict_proba({"data": df})
    pred = np.asarray(pred)
    return pred[:, 1]


def rank_average(pred_list):
    rank_list = [pd.Series(p).rank(method="average").to_numpy() for p in pred_list]
    return np.mean(rank_list, axis=0)


# Leg 1: full train_part
leg_full_df = train_part

# Leg 2: even rows within train_part
leg_even_df = train_part.iloc[np.arange(len(train_part)) % 2 == 0].copy()

# Leg 3: odd rows within train_part
leg_odd_df = train_part.iloc[np.arange(len(train_part)) % 2 == 1].copy()

learner_full = train_one_leg(leg_full_df)
learner_even = train_one_leg(leg_even_df)
learner_odd = train_one_leg(leg_odd_df)

p_full = predict_positive_proba(learner_full, valid_part)
p_even = predict_positive_proba(learner_even, valid_part)
p_odd = predict_positive_proba(learner_odd, valid_part)

weighted_blend = 0.5 * p_full + 0.25 * p_even + 0.25 * p_odd
equal_blend = (p_full + p_even + p_odd) / 3.0
rank_blend = rank_average([p_full, p_even, p_odd])

y_valid = valid_part[target_col].to_numpy()

weighted_score = roc_auc_score(y_valid, weighted_blend)
equal_score = roc_auc_score(y_valid, equal_blend)
rank_score = roc_auc_score(y_valid, rank_blend)

blend_scores = {
    "weighted": weighted_score,
    "equal": equal_score,
    "rank": rank_score,
}

best_blend_name = max(blend_scores, key=blend_scores.get)
final_validation_score = blend_scores[best_blend_name]

print(f"Final Validation Performance: {final_validation_score}")
