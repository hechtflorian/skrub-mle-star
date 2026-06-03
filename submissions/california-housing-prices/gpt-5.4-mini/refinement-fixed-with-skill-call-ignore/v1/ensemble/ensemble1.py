
import os
import warnings

import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_learner(X_df, y_ser, model_kind="hgb"):
    data = skrub.var("data", X_df)
    X = data.skb.mark_as_X()
    y = skrub.var("target", y_ser).skb.mark_as_y()

    vectorizer = skrub.TableVectorizer(
        high_cardinality=skrub.choose_from(
            {
                "gap": skrub.GapEncoder(),
                "minhash": skrub.MinHashEncoder(n_components=8),
            },
            name="high_card_encoder",
        )
    )

    X_vec = X.skb.apply(vectorizer)

    if model_kind == "rf":
        regressor = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_leaf=2,
            random_state=0,
            n_jobs=-1,
        )
    else:
        regressor = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=8,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=42,
        )

    pred_plan = X_vec.skb.apply(regressor, y=y)
    learner = pred_plan.skb.make_learner(fitted=True)
    return learner


def main():
    train_df, test_df = load_data()
    target_col = "median_house_value"

    X_df = train_df.drop(columns=[target_col])
    y_ser = train_df[target_col].copy()

    seeds = [42, 7, 202, 1234, 999]
    seeds = seeds[:5]

    split_hgb_preds = []
    split_rf_preds = []
    split_blended_preds = []
    split_weights = []
    split_rmses = []

    for seed in seeds:
        X_train_df, X_val_df, y_train_ser, y_val_ser = train_test_split(
            X_df, y_ser, test_size=0.2, random_state=seed
        )

        hgb_learner = build_learner(X_train_df, y_train_ser, model_kind="hgb")
        hgb_learner.fit({"data": X_train_df, "target": y_train_ser})
        hgb_val_pred = np.asarray(hgb_learner.predict({"data": X_val_df, "target": y_val_ser}))

        rf_learner = build_learner(X_train_df, y_train_ser, model_kind="rf")
        rf_learner.fit({"data": X_train_df, "target": y_train_ser})
        rf_val_pred = np.asarray(rf_learner.predict({"data": X_val_df, "target": y_val_ser}))

        hgb_rmse = mean_squared_error(y_val_ser, hgb_val_pred) ** 0.5
        rf_rmse = mean_squared_error(y_val_ser, rf_val_pred) ** 0.5

        hgb_w = 1.0 / max(hgb_rmse, 1e-12)
        rf_w = 1.0 / max(rf_rmse, 1e-12)
        w_sum = hgb_w + rf_w
        hgb_w /= w_sum
        rf_w /= w_sum

        blended_val_pred = hgb_w * hgb_val_pred + rf_w * rf_val_pred
        blended_rmse = mean_squared_error(y_val_ser, blended_val_pred) ** 0.5

        split_hgb_preds.append(hgb_val_pred)
        split_rf_preds.append(rf_val_pred)
        split_blended_preds.append(blended_val_pred)
        split_weights.append((hgb_w, rf_w))
        split_rmses.append((hgb_rmse, rf_rmse, blended_rmse))

    mean_hgb_rmse = float(np.mean([x[0] for x in split_rmses]))
    mean_rf_rmse = float(np.mean([x[1] for x in split_rmses]))
    mean_blended_rmse = float(np.mean([x[2] for x in split_rmses]))

    final_validation_score = mean_blended_rmse
    print(f"Final Validation Performance: {final_validation_score}")

    avg_hgb_weight = float(np.mean([w[0] for w in split_weights]))
    avg_rf_weight = float(np.mean([w[1] for w in split_weights]))
    weight_sum = avg_hgb_weight + avg_rf_weight
    avg_hgb_weight /= weight_sum
    avg_rf_weight /= weight_sum

    full_hgb_learner = build_learner(X_df, y_ser, model_kind="hgb")
    full_hgb_learner.fit({"data": X_df, "target": y_ser})
    hgb_test_pred = np.asarray(full_hgb_learner.predict({"data": test_df}))

    full_rf_learner = build_learner(X_df, y_ser, model_kind="rf")
    full_rf_learner.fit({"data": X_df, "target": y_ser})
    rf_test_pred = np.asarray(full_rf_learner.predict({"data": test_df}))

    test_pred = avg_hgb_weight * hgb_test_pred + avg_rf_weight * rf_test_pred

    target_min = float(y_ser.min())
    target_max = float(y_ser.max())
    test_pred = np.clip(test_pred, target_min, target_max)

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
