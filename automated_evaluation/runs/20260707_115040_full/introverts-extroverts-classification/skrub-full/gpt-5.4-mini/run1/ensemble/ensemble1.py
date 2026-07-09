
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Personality"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def add_missing_indicators(df):
    out = df.copy()
    null_cols = out.columns[out.isna().any()]
    for col in null_cols:
        out[f"{col}__is_null"] = out[col].isna().astype("int8")
    return out


def make_branch_route(df):
    out = df.copy()
    cols = [c for c in out.columns if c != target_col]

    # Light column routing split: create a distinct view without changing the backbone.
    # We keep all original non-target columns, but reorder them through a routed split
    # and add a tiny structural difference by placing sparse-ish columns first.
    null_rate = out[cols].isna().mean().sort_values(ascending=False)
    routed_cols = list(null_rate.index)

    # Recombine only non-target columns into this second branch.
    out = out[routed_cols + ([target_col] if target_col in out.columns else [])]
    return out


def get_positive_scores(learner, df):
    try:
        proba = learner.predict_proba({"data": df})
        proba = np.asarray(proba)
        if proba.ndim == 2:
            if proba.shape[1] == 1:
                return proba[:, 0].astype(float)
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)
    except Exception:
        pred = learner.predict({"data": df})
        pred = np.asarray(pred)
        if pred.dtype.kind in {"U", "S", "O"}:
            # map labels to binary consistently
            uniq = pd.unique(pred)
            if len(uniq) == 2:
                pos_label = sorted(map(str, uniq))[-1]
                return (pred.astype(str) == pos_label).astype(float)
            return pd.Series(pred).astype("category").cat.codes.to_numpy(dtype=float)
        return pred.astype(float).ravel()


def to_label_from_score(score, reference_labels=None):
    score = np.asarray(score).ravel()
    labels = (score >= 0.5).astype(int)
    if reference_labels is None:
        return labels
    ref = np.asarray(reference_labels)
    if ref.dtype.kind in {"U", "S", "O"}:
        uniq = pd.unique(ref.astype(str))
        if len(uniq) == 2:
            neg_label, pos_label = sorted(map(str, uniq))
            return np.where(labels == 1, pos_label, neg_label)
        return labels
    return labels


# Shared upstream DataOps root
data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_missing_indicators)

# Leg 1: stabilized pipeline
X_train_1 = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_1 = data_train[target_col].skb.mark_as_y()

cleaner_1 = skrub.Cleaner(drop_if_constant=True)
vectorizer_1 = skrub.TableVectorizer()
model_1 = CatBoostClassifier(
    loss_function="Logloss",
    verbose=0,
    random_seed=random_state,
)

pred_1 = (
    X_train_1.skb.apply(cleaner_1)
    .skb.apply(vectorizer_1)
    .skb.apply(model_1, y=y_train_1)
)
learner_1 = pred_1.skb.make_learner(fitted=True)

# Leg 2: same backbone, but a different feature view via light column routing
data_train_2 = data_train.skb.apply_func(make_branch_route)
X_train_2 = data_train_2.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train_2 = data_train_2[target_col].skb.mark_as_y()

cleaner_2 = skrub.Cleaner(drop_if_constant=True)
vectorizer_2 = skrub.TableVectorizer()
model_2 = CatBoostClassifier(
    loss_function="Logloss",
    verbose=0,
    random_seed=random_state + 7,
)

pred_2 = (
    X_train_2.skb.apply(cleaner_2)
    .skb.apply(vectorizer_2)
    .skb.apply(model_2, y=y_train_2)
)
learner_2 = pred_2.skb.make_learner(fitted=True)

# Blend predictions on the same validation split
p1 = get_positive_scores(learner_1, valid_part)
p2 = get_positive_scores(learner_2, valid_part)

w1, w2 = 0.65, 0.35
soft_blend = w1 * p1 + w2 * p2

# Tiny confidence gate: if disagree and one leg is near boundary, trust stable leg
label_1 = (p1 >= 0.5).astype(int)
label_2 = (p2 >= 0.5).astype(int)
disagree = label_1 != label_2
near_boundary = (np.abs(p1 - 0.5) <= 0.08) | (np.abs(p2 - 0.5) <= 0.08)
gate_mask = disagree & near_boundary
soft_blend[gate_mask] = p1[gate_mask]

valid_pred = to_label_from_score(soft_blend, reference_labels=valid_part[target_col])
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
