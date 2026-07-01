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
    train_df = pd.read_csv(train_path)
    return train_df


def build_learner(
    X_df,
    y_ser,
    model_kind="hgb",
    encoder_kind="gap",
    use_high_cardinality_encoder=True,
):
    data = skrub.var("data", X_df)
    X = data.skb.mark_as_X()
    y = skrub.var("target", y_ser).skb.mark_as_y()

    if use_high_cardinality_encoder:
        if encoder_kind == "minhash":
            high_card_encoder = skrub.MinHashEncoder(n_components=8)
        else:
            high_card_encoder = skrub.GapEncoder()
        vectorizer = skrub.TableVectorizer(high_cardinality=high_card_encoder)
    else:
        vectorizer = skrub.TableVectorizer()

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


def evaluate_variant(X_train_df, y_train_ser, X_val_df, y_val_ser, variant_name, **kwargs):
    learner = build_learner(X_train_df, y_train_ser, **kwargs)
    learner.fit({"data": X_train_df, "target": y_train_ser})
    val_pred = learner.predict({"data": X_val_df, "target": y_val_ser})
    rmse = mean_squared_error(y_val_ser, val_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {rmse}")
    return rmse


def main():
    train_df = load_data()
    target_col = "median_house_value"

    X_df = train_df.drop(columns=[target_col])
    y_ser = train_df[target_col].copy()

    X_train_df, X_val_df, y_train_ser, y_val_ser = train_test_split(
        X_df, y_ser, test_size=0.2, random_state=42
    )

    results = {}

    results["baseline_hgb_gap"] = evaluate_variant(
        X_train_df,
        y_train_ser,
        X_val_df,
        y_val_ser,
        variant_name="baseline_hgb_gap",
        model_kind="hgb",
        encoder_kind="gap",
        use_high_cardinality_encoder=True,
    )

    results["no_high_card_encoder_hgb"] = evaluate_variant(
        X_train_df,
        y_train_ser,
        X_val_df,
        y_val_ser,
        variant_name="no_high_card_encoder_hgb",
        model_kind="hgb",
        encoder_kind="gap",
        use_high_cardinality_encoder=False,
    )

    results["baseline_rf_gap"] = evaluate_variant(
        X_train_df,
        y_train_ser,
        X_val_df,
        y_val_ser,
        variant_name="baseline_rf_gap",
        model_kind="rf",
        encoder_kind="gap",
        use_high_cardinality_encoder=True,
    )

    baseline = results["baseline_hgb_gap"]
    comparisons = {
        "no_high_card_encoder_hgb": results["no_high_card_encoder_hgb"] - baseline,
        "baseline_rf_gap": results["baseline_rf_gap"] - baseline,
    }

    print(f"Final Validation Performance: {baseline}")

    best_variant = min(results, key=results.get)
    best_score = results[best_variant]

    print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")

    most_important_variant = max(comparisons, key=lambda k: abs(comparisons[k]))
    direction = "worse" if comparisons[most_important_variant] > 0 else "better"
    print(
        f"Most impactful change: {most_important_variant} | "
        f"Delta RMSE vs baseline: {comparisons[most_important_variant]} ({direction})"
    )


if __name__ == "__main__":
    main()