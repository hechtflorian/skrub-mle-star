
import os
import glob
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

random_state = 42
target_col = "Rings"

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred = np.clip(y_pred, 0, None)
    return mean_squared_log_error(y_true, y_pred) ** 0.5

def find_csv(name):
    candidates = [
        os.path.join(".", "input", name),
        os.path.join(".", "input", "*", name),
    ]
    for pattern in candidates:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find {name} under ./input")

def build_predictor(train_subset):
    data_train = skrub.var("data", train_subset)

    X_train = (
        data_train
        .drop(columns=[target_col], errors="ignore")
        .skb.mark_as_X()
        .skb.apply(skrub.DropCols(cols=["id"]))
    )
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        random_state=random_state,
        verbose=-1
    )

    predictor = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    return predictor

train_path = find_csv("train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# Leg 1: original full train split
predictor_1 = build_predictor(train_part)
learner_1 = predictor_1.skb.make_learner(fitted=True)
pred_1 = np.clip(np.asarray(learner_1.predict({"data": valid_part})), 0, None)

# Leg 2: bootstrap-like resample (95% rows, without replacement)
rng = np.random.RandomState(random_state)
leg2_idx = rng.choice(len(train_part), size=max(1, int(0.95 * len(train_part))), replace=False)
train_part_leg2 = train_part.iloc[leg2_idx].copy()

predictor_2 = build_predictor(train_part_leg2)
learner_2 = predictor_2.skb.make_learner(fitted=True)
pred_2 = np.clip(np.asarray(learner_2.predict({"data": valid_part})), 0, None)

# Leg 3: differently sampled subset (complement of an 85% sample)
rng3 = np.random.RandomState(random_state + 1)
sample3_idx = rng3.choice(len(train_part), size=max(1, int(0.85 * len(train_part))), replace=False)
mask3 = np.ones(len(train_part), dtype=bool)
mask3[sample3_idx] = False
comp_idx = np.where(mask3)[0]

if len(comp_idx) == 0:
    extra_idx = rng3.choice(len(train_part), size=max(1, int(0.15 * len(train_part))), replace=False)
    train_part_leg3 = train_part.iloc[extra_idx].copy()
else:
    train_part_leg3 = train_part.iloc[comp_idx].copy()

predictor_3 = build_predictor(train_part_leg3)
learner_3 = predictor_3.skb.make_learner(fitted=True)
pred_3 = np.clip(np.asarray(learner_3.predict({"data": valid_part})), 0, None)

# Compare a few preset merge rules on holdout, then keep the best
weight_patterns = [
    [1.0, 0.0, 0.0],
    [0.7, 0.3, 0.0],
    [0.5, 0.25, 0.25],
]

best_score = None
best_pred = None

for w1, w2, w3 in weight_patterns:
    blend = w1 * pred_1 + w2 * pred_2 + w3 * pred_3
    blend = 0.8 * blend + 0.2 * pred_1
    blend = np.clip(blend, 0, None)
    score = rmsle(valid_part[target_col].to_numpy(), blend)
    if best_score is None or score < best_score:
        best_score = score
        best_pred = blend

final_validation_score = rmsle(valid_part[target_col].to_numpy(), best_pred)
print(f"Final Validation Performance: {final_validation_score}")
