
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "Personality"
train_path = "./input/train.csv"

train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def build_learner(leg_train_df):
    data_train = skrub.var("data", leg_train_df)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    predictor = X_train.skb.apply(vectorizer).skb.apply(
        CatBoostClassifier(
            random_state=random_state,
            verbose=0,
        ),
        y=y_train,
    )
    return predictor.skb.make_learner(fitted=True)


def get_probabilities(learner, df):
    proba = np.asarray(learner.predict_proba({"data": df}))
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    return proba


# Leg 1: original train_part exactly as-is
leg1_train = train_part.copy()

# Leg 2: bootstrap resample with replacement
leg2_train = train_part.sample(
    n=len(train_part),
    replace=True,
    random_state=random_state,
).reset_index(drop=True)

# Leg 3: 90% random subsample without replacement
leg3_train = train_part.sample(
    frac=0.9,
    replace=False,
    random_state=random_state + 1,
).reset_index(drop=True)

learner_1 = build_learner(leg1_train)
learner_2 = build_learner(leg2_train)
learner_3 = build_learner(leg3_train)

proba_1 = get_probabilities(learner_1, valid_part)
proba_2 = get_probabilities(learner_2, valid_part)
proba_3 = get_probabilities(learner_3, valid_part)

avg_proba = (proba_1 + proba_2 + proba_3) / 3.0
classes = np.asarray(learner_1.classes_)
valid_pred = classes[np.argmax(avg_proba, axis=1)]

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
