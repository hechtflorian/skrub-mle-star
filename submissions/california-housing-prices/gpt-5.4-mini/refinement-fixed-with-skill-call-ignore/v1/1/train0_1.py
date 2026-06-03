
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

    X_train_df, X_val_df, y_train_ser, y_val_ser = train_test_split(
        X_df, y_ser, test_size=0.2, random_state=42
    )

    hgb_learner = build_learner(X_train_df, y_train_ser, model_kind="hgb")
    hgb_learner.fit({"data": X_train_df, "target": y_train_ser})
    hgb_val_pred = hgb_learner.predict({"data": X_val_df, "target": y_val_ser})
    hgb_rmse = mean_squared_error(y_val_ser, hgb_val_pred) ** 0.5
    print(f"HGB Validation Performance: {hgb_rmse}")

    rf_learner = build_learner(X_train_df, y_train_ser, model_kind="rf")
    rf_learner.fit({"data": X_train_df, "target": y_train_ser})
    rf_val_pred = rf_learner.predict({"data": X_val_df, "target": y_val_ser})
    rf_rmse = mean_squared_error(y_val_ser, rf_val_pred) ** 0.5
    print(f"RF Validation Performance: {rf_rmse}")

    final_validation_score = min(hgb_rmse, rf_rmse)
    print(f"Final Validation Performance: {final_validation_score}")

    full_hgb_learner = build_learner(X_df, y_ser, model_kind="hgb")
    full_hgb_learner.fit({"data": X_df, "target": y_ser})
    hgb_test_pred = full_hgb_learner.predict({"data": test_df})

    full_rf_learner = build_learner(X_df, y_ser, model_kind="rf")
    full_rf_learner.fit({"data": X_df, "target": y_ser})
    rf_test_pred = full_rf_learner.predict({"data": test_df})

    test_pred = 0.5 * np.asarray(hgb_test_pred) + 0.5 * np.asarray(rf_test_pred)

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
