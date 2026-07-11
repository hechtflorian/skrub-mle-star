
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


def make_bootstrap_sample(df, seed):
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(df), size=len(df), replace=True)
    return df.iloc[sample_idx].copy()


def make_stratified_subsample(df, target_col, frac, seed):
    parts = []
    for _, group in df.groupby(target_col, group_keys=False):
        n_take = max(1, int(round(len(group) * frac)))
        n_take = min(n_take, len(group))
        parts.append(group.sample(n=n_take, replace=False, random_state=seed))
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def align_proba(proba, learner_classes, class_order):
    class_to_idx = {c: i for i, c in enumerate(learner_classes)}
    aligned = np.zeros((proba.shape[0], len(class_order)), dtype=float)
    for j, cls in enumerate(class_order):
        aligned[:, j] = proba[:, class_to_idx[cls]]
    return aligned


def proba_to_rank_scores(proba):
    n_rows, n_classes = proba.shape
    if n_classes == 1:
        return np.ones_like(proba, dtype=float)

    order = np.argsort(proba, axis=1)
    ranks = np.empty_like(order, dtype=int)
    row_idx = np.arange(n_rows)[:, None]
    ranks[row_idx, order] = np.arange(n_classes)
    rank_scores = ranks / float(n_classes - 1)
    return rank_scores.astype(float)


def weighted_average(arrays, weights):
    result = np.zeros_like(arrays[0], dtype=float)
    for arr, w in zip(arrays, weights):
        result += w * arr
    return result


def build_leg(train_df_leg, target_col):
    data_train_leg = skrub.var("data", train_df_leg)
    predictor = predictor_lgbm(data_train_leg, target_col)
    learner = predictor.skb.make_learner(fitted=True)
    return learner


def get_aligned_leg_outputs(learner, valid_part, class_order):
    proba = np.asarray(learner.predict_proba({"data": valid_part}))
    learner_classes = list(learner.classes_)
    aligned_proba = align_proba(proba, learner_classes, class_order)
    aligned_rank = proba_to_rank_scores(aligned_proba)
    return aligned_proba, aligned_rank


def evaluate_blend(prob_list, rank_list, weights, alpha, class_order, y_true, full_leg_proba=None, gate_threshold=None):
    avg_proba = weighted_average(prob_list, weights)
    avg_rank = weighted_average(rank_list, weights)
    final_score = alpha * avg_proba + (1.0 - alpha) * avg_rank

    if full_leg_proba is not None and gate_threshold is not None:
        confident_mask = np.max(full_leg_proba, axis=1) >= gate_threshold
        final_score[confident_mask] = full_leg_proba[confident_mask]

    pred_idx = np.argmax(final_score, axis=1)
    pred_labels = np.asarray(class_order)[pred_idx]
    return accuracy_score(y_true, pred_labels)


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

    class_order = sorted(train_part[target_col].unique().tolist())

    leg1_train = train_part
    leg2_train = make_bootstrap_sample(train_part, seed=101)
    leg3_train = make_stratified_subsample(train_part, target_col=target_col, frac=0.85, seed=202)
    leg4_train = make_bootstrap_sample(train_part, seed=303)

    learner1 = build_leg(leg1_train, target_col)
    learner2 = build_leg(leg2_train, target_col)
    learner3 = build_leg(leg3_train, target_col)
    learner4 = build_leg(leg4_train, target_col)

    leg1_proba, leg1_rank = get_aligned_leg_outputs(learner1, valid_part, class_order)
    leg2_proba, leg2_rank = get_aligned_leg_outputs(learner2, valid_part, class_order)
    leg3_proba, leg3_rank = get_aligned_leg_outputs(learner3, valid_part, class_order)
    leg4_proba, leg4_rank = get_aligned_leg_outputs(learner4, valid_part, class_order)

    prob_list = [leg1_proba, leg2_proba, leg3_proba, leg4_proba]
    rank_list = [leg1_rank, leg2_rank, leg3_rank, leg4_rank]
    weight_presets = [
        [0.4, 0.2, 0.2, 0.2],
        [0.5, 1 / 6, 1 / 6, 1 / 6],
    ]
    alpha_list = [0.6, 0.7, 0.8]
    gate_options = [None, 0.75]

    best_score = -1.0
    best_config = None

    y_true = valid_part[target_col].values

    for weights in weight_presets:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        for alpha in alpha_list:
            for gate_threshold in gate_options:
                score = evaluate_blend(
                    prob_list=prob_list,
                    rank_list=rank_list,
                    weights=weights,
                    alpha=alpha,
                    class_order=class_order,
                    y_true=y_true,
                    full_leg_proba=leg1_proba,
                    gate_threshold=gate_threshold,
                )
                if score > best_score:
                    best_score = score
                    best_config = (weights, alpha, gate_threshold)

    weights, alpha, gate_threshold = best_config
    avg_proba = weighted_average(prob_list, weights)
    avg_rank = weighted_average(rank_list, weights)
    final_score = alpha * avg_proba + (1.0 - alpha) * avg_rank

    if gate_threshold is not None:
        confident_mask = np.max(leg1_proba, axis=1) >= gate_threshold
        final_score[confident_mask] = leg1_proba[confident_mask]

    final_pred_idx = np.argmax(final_score, axis=1)
    final_pred = np.asarray(class_order)[final_pred_idx]

    final_validation_score = accuracy_score(valid_part[target_col], final_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    test_path = os.path.join(".", "input", "test.csv")
    test_df = pd.read_csv(test_path)

    class_order = sorted(train_df[target_col].unique().tolist())

    leg1_train = train_df
    leg2_train = make_bootstrap_sample(train_df, seed=101)
    leg3_train = make_stratified_subsample(train_df, target_col=target_col, frac=0.85, seed=202)
    leg4_train = make_bootstrap_sample(train_df, seed=303)

    learner1 = build_leg(leg1_train, target_col)
    learner2 = build_leg(leg2_train, target_col)
    learner3 = build_leg(leg3_train, target_col)
    learner4 = build_leg(leg4_train, target_col)

    leg1_proba, leg1_rank = get_aligned_leg_outputs(learner1, test_df, class_order)
    leg2_proba, leg2_rank = get_aligned_leg_outputs(learner2, test_df, class_order)
    leg3_proba, leg3_rank = get_aligned_leg_outputs(learner3, test_df, class_order)
    leg4_proba, leg4_rank = get_aligned_leg_outputs(learner4, test_df, class_order)

    prob_list = [leg1_proba, leg2_proba, leg3_proba, leg4_proba]
    rank_list = [leg1_rank, leg2_rank, leg3_rank, leg4_rank]

    avg_proba = weighted_average(prob_list, weights)
    avg_rank = weighted_average(rank_list, weights)
    final_score = alpha * avg_proba + (1.0 - alpha) * avg_rank

    if gate_threshold is not None:
        confident_mask = np.max(leg1_proba, axis=1) >= gate_threshold
        final_score[confident_mask] = leg1_proba[confident_mask]

    test_pred_idx = np.argmax(final_score, axis=1)
    test_pred = np.asarray(class_order)[test_pred_idx]

    os.makedirs("./final", exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            target_col: test_pred,
        }
    )
    submission.to_csv("./final/submission.csv", index=False)


if __name__ == "__main__":
    main()
