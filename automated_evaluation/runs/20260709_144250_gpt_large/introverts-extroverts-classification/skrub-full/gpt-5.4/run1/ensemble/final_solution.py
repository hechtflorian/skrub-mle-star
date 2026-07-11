
import warnings

warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "Personality"
agreement_threshold = 0.80

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


def build_learner(train_subset):
    data_train = skrub.var("data", train_subset)
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
    return np.asarray(learner.predict_proba({"data": df}))


# Leg 1: original anchor model unchanged
learner_1 = build_learner(train_part)

# Leg 2: bootstrap-resampled train_part
rng_bootstrap = np.random.RandomState(random_state + 1)
bootstrap_indices = rng_bootstrap.choice(len(train_part), size=len(train_part), replace=True)
train_part_bootstrap = train_part.iloc[bootstrap_indices].copy().reset_index(drop=True)
learner_2 = build_learner(train_part_bootstrap)

# Leg 3: complementary subsample from rows not chosen by a random 87.5% subsample
rng_complement = np.random.RandomState(random_state + 2)
subsample_frac = 0.875
chosen_size = max(1, int(len(train_part) * subsample_frac))
chosen_indices = rng_complement.choice(len(train_part), size=chosen_size, replace=False)
mask = np.ones(len(train_part), dtype=bool)
mask[chosen_indices] = False
complement_indices = np.where(mask)[0]
if len(complement_indices) == 0:
    complement_indices = np.arange(len(train_part))
train_part_complement = train_part.iloc[complement_indices].copy().reset_index(drop=True)
learner_3 = build_learner(train_part_complement)

# Validation probabilities
proba_1 = get_probabilities(learner_1, valid_part)
proba_2 = get_probabilities(learner_2, valid_part)
proba_3 = get_probabilities(learner_3, valid_part)

classes = np.asarray(learner_1.classes_)

# Per-leg predictions and accuracies
pred_1 = classes[np.argmax(proba_1, axis=1)]
pred_2 = classes[np.argmax(proba_2, axis=1)]
pred_3 = classes[np.argmax(proba_3, axis=1)]

acc_1 = accuracy_score(valid_part[target_col], pred_1)
acc_2 = accuracy_score(valid_part[target_col], pred_2)
acc_3 = accuracy_score(valid_part[target_col], pred_3)

weights = np.array([acc_1**3, acc_2**3, acc_3**3], dtype=float)
if weights.sum() == 0:
    weights = np.array([1.0, 1.0, 1.0], dtype=float)
weights = weights / weights.sum()

# Weighted soft blend
blend = weights[0] * proba_1 + weights[1] * proba_2 + weights[2] * proba_3
blend_pred = classes[np.argmax(blend, axis=1)]

# Agreement gate
conf_1 = np.max(proba_1, axis=1)
conf_2 = np.max(proba_2, axis=1)
conf_3 = np.max(proba_3, axis=1)

final_pred = blend_pred.copy()

for i in range(len(valid_part)):
    row_preds = [pred_1[i], pred_2[i], pred_3[i]]
    row_confs = [conf_1[i], conf_2[i], conf_3[i]]

    agreed_label = None

    if (
        row_preds[0] == row_preds[1]
        and row_confs[0] >= agreement_threshold
        and row_confs[1] >= agreement_threshold
        and row_preds[0] == pred_1[i]
    ):
        agreed_label = row_preds[0]
    elif (
        row_preds[0] == row_preds[2]
        and row_confs[0] >= agreement_threshold
        and row_confs[2] >= agreement_threshold
        and row_preds[0] == pred_1[i]
    ):
        agreed_label = row_preds[0]
    elif (
        row_preds[1] == row_preds[2]
        and row_confs[1] >= agreement_threshold
        and row_confs[2] >= agreement_threshold
        and row_preds[1] == pred_1[i]
    ):
        agreed_label = row_preds[1]

    if agreed_label is not None:
        final_pred[i] = agreed_label

final_validation_score = accuracy_score(valid_part[target_col], final_pred)
print(f"Final Validation Performance: {final_validation_score}")

test_path = "./input/test.csv"
test_df = pd.read_csv(test_path)

full_learner_1 = build_learner(train_df)

rng_bootstrap_full = np.random.RandomState(random_state + 1)
bootstrap_indices_full = rng_bootstrap_full.choice(len(train_df), size=len(train_df), replace=True)
train_df_bootstrap = train_df.iloc[bootstrap_indices_full].copy().reset_index(drop=True)
full_learner_2 = build_learner(train_df_bootstrap)

rng_complement_full = np.random.RandomState(random_state + 2)
chosen_size_full = max(1, int(len(train_df) * subsample_frac))
chosen_indices_full = rng_complement_full.choice(len(train_df), size=chosen_size_full, replace=False)
mask_full = np.ones(len(train_df), dtype=bool)
mask_full[chosen_indices_full] = False
complement_indices_full = np.where(mask_full)[0]
if len(complement_indices_full) == 0:
    complement_indices_full = np.arange(len(train_df))
train_df_complement = train_df.iloc[complement_indices_full].copy().reset_index(drop=True)
full_learner_3 = build_learner(train_df_complement)

test_proba_1 = get_probabilities(full_learner_1, test_df)
test_proba_2 = get_probabilities(full_learner_2, test_df)
test_proba_3 = get_probabilities(full_learner_3, test_df)

test_classes = np.asarray(full_learner_1.classes_)

test_pred_1 = test_classes[np.argmax(test_proba_1, axis=1)]
test_pred_2 = test_classes[np.argmax(test_proba_2, axis=1)]
test_pred_3 = test_classes[np.argmax(test_proba_3, axis=1)]

test_blend = weights[0] * test_proba_1 + weights[1] * test_proba_2 + weights[2] * test_proba_3
test_blend_pred = test_classes[np.argmax(test_blend, axis=1)]

test_conf_1 = np.max(test_proba_1, axis=1)
test_conf_2 = np.max(test_proba_2, axis=1)
test_conf_3 = np.max(test_proba_3, axis=1)

test_final_pred = test_blend_pred.copy()

for i in range(len(test_df)):
    row_preds = [test_pred_1[i], test_pred_2[i], test_pred_3[i]]
    row_confs = [test_conf_1[i], test_conf_2[i], test_conf_3[i]]

    agreed_label = None

    if (
        row_preds[0] == row_preds[1]
        and row_confs[0] >= agreement_threshold
        and row_confs[1] >= agreement_threshold
        and row_preds[0] == test_pred_1[i]
    ):
        agreed_label = row_preds[0]
    elif (
        row_preds[0] == row_preds[2]
        and row_confs[0] >= agreement_threshold
        and row_confs[2] >= agreement_threshold
        and row_preds[0] == test_pred_1[i]
    ):
        agreed_label = row_preds[0]
    elif (
        row_preds[1] == row_preds[2]
        and row_confs[1] >= agreement_threshold
        and row_confs[2] >= agreement_threshold
        and row_preds[1] == test_pred_1[i]
    ):
        agreed_label = row_preds[1]

    if agreed_label is not None:
        test_final_pred[i] = agreed_label

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame(
    {
        "id": test_df["id"],
        target_col: test_final_pred,
    }
)
submission.to_csv("./final/submission.csv", index=False)
