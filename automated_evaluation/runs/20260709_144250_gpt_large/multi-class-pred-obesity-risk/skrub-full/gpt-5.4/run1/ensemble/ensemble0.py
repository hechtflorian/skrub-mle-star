
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def predictor_lgbm(data_train, target_col):
    X_train = (
        data_train.drop(columns=["id"], errors="ignore")
        .drop(columns=target_col, errors="ignore")
        .skb.mark_as_X()
    )
    y_train = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    predictor = X_train.skb.apply(vectorizer).skb.apply(
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        ),
        y=y_train,
    )
    return predictor


def build_learner(train_slice, target_col):
    data_train = skrub.var("data", train_slice)
    predictor = predictor_lgbm(data_train, target_col)
    learner = predictor.skb.make_learner(fitted=True)
    return learner


def bootstrap_sample(df, random_state):
    return df.sample(n=len(df), replace=True, random_state=random_state).reset_index(drop=True)


def stratified_subsample(df, target_col, frac, random_state):
    parts = []
    for _, group in df.groupby(target_col, group_keys=False):
        n_take = max(1, int(round(len(group) * frac)))
        n_take = min(n_take, len(group))
        parts.append(group.sample(n=n_take, replace=False, random_state=random_state))
    sampled = pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def predict_proba_aligned(learner, df, class_order):
    proba = np.asarray(learner.predict_proba({"data": df}))
    learner_classes = np.asarray(learner.classes_)
    aligned = np.zeros((len(df), len(class_order)), dtype=float)
    class_to_idx = {cls: i for i, cls in enumerate(learner_classes)}
    for j, cls in enumerate(class_order):
        if cls in class_to_idx:
            aligned[:, j] = proba[:, class_to_idx[cls]]
    return aligned


def main():
    train_path = os.path.join(".", "input", "train.csv")
    train_df = pd.read_csv(train_path)

    target_col = "NObeyesdad"
    random_state = 42

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=0.2,
        random_state=random_state,
        stratify=train_df[target_col],
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    train_part_full = train_part
    train_part_boot = bootstrap_sample(train_part, random_state=random_state + 1)
    train_part_sub = stratified_subsample(
        train_part, target_col=target_col, frac=0.9, random_state=random_state + 2
    )

    learner_full = build_learner(train_part_full, target_col)
    learner_boot = build_learner(train_part_boot, target_col)
    learner_sub = build_learner(train_part_sub, target_col)

    class_order = np.asarray(learner_full.classes_)

    valid_proba_full = predict_proba_aligned(learner_full, valid_part, class_order)
    valid_proba_boot = predict_proba_aligned(learner_boot, valid_part, class_order)
    valid_proba_sub = predict_proba_aligned(learner_sub, valid_part, class_order)

    weights = np.array([0.5, 0.25, 0.25], dtype=float)
    valid_proba_blend = (
        weights[0] * valid_proba_full
        + weights[1] * valid_proba_boot
        + weights[2] * valid_proba_sub
    )

    valid_pred = class_order[np.argmax(valid_proba_blend, axis=1)]

    final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
    print(f"Final Validation Performance: {final_validation_score}")


if __name__ == "__main__":
    main()
