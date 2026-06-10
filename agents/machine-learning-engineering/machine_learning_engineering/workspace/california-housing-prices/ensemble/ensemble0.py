
import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

TARGET = "median_house_value"
RANDOM_STATE = 42

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Ensemble configuration: keep the original skrub + CatBoost pipeline intact,
# but repeat it across multiple folds and seeds, then average predictions.
n_splits = 5
seed_list = [42, 52, 62]

kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

fold_valid_preds = []
fold_test_preds = []
fold_rmse_scores = []

for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    seed_valid_preds = []
    seed_test_preds = []

    for seed in seed_list:
        # Block 1: honest holdout validation on this fold
        data_train = skrub.var("data", train_part)
        X_train = data_train.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
        y_train = data_train[TARGET].skb.mark_as_y()

        vectorizer = skrub.TableVectorizer(
            low_cardinality="passthrough",
            high_cardinality=skrub.StringEncoder(n_components=20),
        )

        model = CatBoostRegressor(
            loss_function="RMSE",
            depth=8,
            learning_rate=0.03,
            iterations=5000,
            random_seed=seed,
            verbose=200,
        )

        pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
        val_learner = pred_graph.skb.make_learner(fitted=True)

        valid_pred = val_learner.predict({"data": valid_part})
        seed_valid_preds.append(valid_pred)

        # Block 2: full train for test predictions for this seed/fold
        data_full = skrub.var("data", train_df)
        X_full = data_full.drop(columns=TARGET, errors="ignore").skb.mark_as_X()
        y_full = data_full[TARGET].skb.mark_as_y()

        full_pred_graph = X_full.skb.apply(vectorizer).skb.apply(model, y=y_full)
        full_learner = full_pred_graph.skb.make_learner(fitted=True)

        test_pred = full_learner.predict({"data": test_df})
        seed_test_preds.append(test_pred)

    # Average across seeds within the fold
    fold_valid_pred = np.mean(np.vstack(seed_valid_preds), axis=0)
    fold_test_pred = np.mean(np.vstack(seed_test_preds), axis=0)

    fold_rmse = mean_squared_error(valid_part[TARGET], fold_valid_pred) ** 0.5
    print(f"Fold {fold_idx} Validation Performance: {fold_rmse}")

    fold_valid_preds.append((valid_idx, fold_valid_pred))
    fold_test_preds.append(fold_test_pred)
    fold_rmse_scores.append(fold_rmse)

# Overall validation performance computed from concatenated OOF predictions
oof_pred = np.zeros(len(train_df), dtype=float)
for valid_idx, valid_pred in fold_valid_preds:
    oof_pred[valid_idx] = valid_pred

final_validation_score = mean_squared_error(train_df[TARGET], oof_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Weighted average across folds for test prediction, weights inversely proportional to fold RMSE
fold_rmse_scores = np.array(fold_rmse_scores, dtype=float)
fold_weights = 1.0 / np.maximum(fold_rmse_scores, 1e-12)
fold_weights = fold_weights / fold_weights.sum()

test_pred = np.sum(np.vstack(fold_test_preds).T * fold_weights, axis=1)

submission = pd.DataFrame({TARGET: test_pred})
submission.to_csv("submission.csv", index=False)
