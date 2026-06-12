import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

target_col = "median_house_value"
train_path = "./input/train.csv"

train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


@skrub.deferred
def add_house_feature_ratios(df):
    out = df.copy()
    if "total_rooms" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "population" in out.columns and "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        out["population_per_household"] = (out["population"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    if "total_bedrooms" in out.columns and "total_rooms" in out.columns:
        denom = out["total_rooms"].replace(0, np.nan)
        out["bedrooms_per_room"] = (out["total_bedrooms"] / denom).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    return out


def evaluate_variant(variant_name, feature_fn, vectorizer, model):
    data_train = skrub.var("data", train_part)
    data_train_fe = data_train.skb.apply_func(feature_fn)

    X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train_fe[target_col].skb.mark_as_y()

    pred_graph = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
    val_learner = pred_graph.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict({"data": valid_part})

    rmse = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    print(f"Ablation[{variant_name}] RMSE: {rmse}")
    return rmse


def main():
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )

    variants = {
        "baseline_all_ratios": add_house_feature_ratios,
        "no_derived_ratios": skrub.deferred(lambda df: df.copy()),
        "drop_households_only": skrub.deferred(
            lambda df: df.drop(columns=["households"], errors="ignore")
        ),
    }

    results = {}

    for name, feat_fn in variants.items():
        vectorizer = skrub.TableVectorizer()
        results[name] = evaluate_variant(name, feat_fn, vectorizer, model)

    best_variant = min(results, key=results.get)
    best_score = results[best_variant]

    baseline = results["baseline_all_ratios"]
    print(f"Baseline RMSE: {baseline}")
    print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")

    deltas = {
        name: baseline - score for name, score in results.items() if name != "baseline_all_ratios"
    }
    most_important_variant = max(deltas, key=lambda k: abs(deltas[k]))
    most_important_delta = deltas[most_important_variant]

    print(
        f"Most influential change: {most_important_variant} "
        f"(RMSE change vs baseline: {most_important_delta:+.6f})"
    )
    print(f"Final Validation Performance: {baseline}")


if __name__ == "__main__":
    main()