
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def main():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "median_house_value"

    # Stable split for validation
    rng = np.random.RandomState(42)
    idx = np.arange(len(train_df))
    rng.shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx = idx[:split]
    valid_idx = idx[split:]

    train_split = train_df.iloc[train_idx].reset_index(drop=True)
    valid_split = train_df.iloc[valid_idx].reset_index(drop=True)

    # -------------------------
    # Model 1: tuned DataOps HGB
    # -------------------------
    data = skrub.var("data", train_split)
    X = data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    vectorizer_1 = skrub.TableVectorizer(
        high_cardinality=skrub.choose_from(
            {
                "minhash": skrub.MinHashEncoder(n_components=16),
                "gap": skrub.GapEncoder(n_components=16),
            },
            name="high_card_encoder",
        )
    )

    model_1 = HistGradientBoostingRegressor(
        learning_rate=skrub.choose_float(0.03, 0.12, log=True, name="learning_rate"),
        max_depth=skrub.choose_int(4, 8, name="max_depth"),
        min_samples_leaf=skrub.choose_int(15, 50, name="min_samples_leaf"),
        random_state=42,
    )

    pred_1 = X.skb.apply(vectorizer_1).skb.apply(model_1, y=y)
    learner_1 = pred_1.skb.make_learner(fitted=True)

    valid_features = valid_split.drop(columns=[target_col], errors="ignore")
    valid_pred_1 = np.asarray(learner_1.predict({"data": valid_features}))
    score_1 = rmse(valid_split[target_col].to_numpy(), valid_pred_1)

    # -------------------------
    # Model 2: reference-style simple baseline
    # -------------------------
    data_2 = skrub.var("data", train_split)
    X_2 = data_2.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    y_2 = data_2[target_col].skb.mark_as_y()

    vectorizer_2 = skrub.TableVectorizer()
    model_2 = HistGradientBoostingRegressor(random_state=42)

    pred_2 = X_2.skb.apply(vectorizer_2).skb.apply(model_2, y=y_2)
    learner_2 = pred_2.skb.make_learner(fitted=True)

    valid_pred_2 = np.asarray(learner_2.predict({"data": valid_features}))
    score_2 = rmse(valid_split[target_col].to_numpy(), valid_pred_2)

    # Simple ensemble of the two models
    ensemble_valid_pred = 0.5 * valid_pred_1 + 0.5 * valid_pred_2
    final_validation_score = rmse(valid_split[target_col].to_numpy(), ensemble_valid_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    # Fit final ensemble members on full training data and predict test
    full_data = skrub.var("data", train_df)
    full_X = full_data.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
    full_y = full_data[target_col].skb.mark_as_y()

    full_pred_1 = full_X.skb.apply(
        skrub.TableVectorizer(
            high_cardinality=skrub.choose_from(
                {
                    "minhash": skrub.MinHashEncoder(n_components=16),
                    "gap": skrub.GapEncoder(n_components=16),
                },
                name="high_card_encoder",
            )
        )
    ).skb.apply(
        HistGradientBoostingRegressor(
            learning_rate=skrub.choose_float(0.03, 0.12, log=True, name="learning_rate"),
            max_depth=skrub.choose_int(4, 8, name="max_depth"),
            min_samples_leaf=skrub.choose_int(15, 50, name="min_samples_leaf"),
            random_state=42,
        ),
        y=full_y,
    )
    full_learner_1 = full_pred_1.skb.make_learner(fitted=True)

    full_pred_2 = full_X.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=42),
        y=full_y,
    )
    full_learner_2 = full_pred_2.skb.make_learner(fitted=True)

    test_pred_1 = np.asarray(full_learner_1.predict({"data": test_df}))
    test_pred_2 = np.asarray(full_learner_2.predict({"data": test_df}))
    test_preds = 0.5 * test_pred_1 + 0.5 * test_pred_2

    submission = pd.DataFrame({target_col: test_preds})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
