
import math
import numpy as np
import pandas as pd
import skrub
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


TARGET_COL = "revenue"
RANDOM_STATE = 42
METRIC_LABEL = "RMSE"


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    open_date = pd.to_datetime(out["Open Date"], format="%m/%d/%Y")
    ref_date = pd.Timestamp("2015-01-01")
    age_days = (ref_date - open_date).dt.days
    out["restaurant_age_days"] = age_days
    out["restaurant_age_log"] = age_days.clip(lower=1).map(math.log)
    out = out.drop(columns=["Open Date"])
    return out


def drop_id_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop(columns=["Id"], errors="ignore")
    return out


def build_model():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbosity=0,
    )


def score_variant(variant_name, train_part, valid_part, fe_func=None, use_id=False):
    data_train = skrub.var("data", train_part)

    if fe_func is not None:
        data_train = data_train.skb.apply_func(fe_func)

    drop_cols = [TARGET_COL]
    if not use_id:
        drop_cols.append("Id")

    X_train = data_train.drop(columns=drop_cols, errors="ignore").skb.mark_as_X()
    y_train = data_train[TARGET_COL].skb.mark_as_y()

    predictor = (
        X_train.skb.apply(skrub.TableVectorizer())
        .skb.apply(build_model(), y=y_train)
    )

    learner = predictor.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = rmse(valid_part[TARGET_COL], valid_pred)
    print(f"Ablation[{variant_name}] {METRIC_LABEL}: {score}")
    return score


def main():
    train_df = pd.read_csv("./input/train.csv")

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.2, random_state=RANDOM_STATE
    )
    train_part = train_df.iloc[train_idx].copy()
    valid_part = train_df.iloc[valid_idx].copy()

    valid_part_baseline = add_features(valid_part)
    valid_part_no_fe = valid_part.copy()
    valid_part_keep_id = add_features(valid_part)

    scores = {}
    scores["baseline"] = score_variant(
        "baseline",
        train_part=add_features(train_part),
        valid_part=valid_part_baseline,
        fe_func=None,
        use_id=False,
    )
    scores["no_age_features"] = score_variant(
        "no_age_features",
        train_part=train_part.copy(),
        valid_part=valid_part_no_fe,
        fe_func=None,
        use_id=False,
    )
    scores["keep_id"] = score_variant(
        "keep_id",
        train_part=add_features(train_part),
        valid_part=valid_part_keep_id,
        fe_func=None,
        use_id=True,
    )

    best_variant = min(scores, key=scores.get)
    best_score = scores[best_variant]
    print(f"Best ablation variant: {best_variant} | {METRIC_LABEL}: {best_score}")

    baseline_score = scores["baseline"]
    if len(scores) > 1:
        most_impact_variant = max(
            (name for name in scores if name != "baseline"),
            key=lambda name: abs(scores[name] - baseline_score),
        )
        impact = scores[most_impact_variant] - baseline_score
        direction = "improves" if impact < 0 else "hurts"
        print(
            f"Part with largest impact vs baseline: {most_impact_variant} | "
            f"delta_{METRIC_LABEL}: {impact} ({direction})"
        )


if __name__ == "__main__":
    main()
