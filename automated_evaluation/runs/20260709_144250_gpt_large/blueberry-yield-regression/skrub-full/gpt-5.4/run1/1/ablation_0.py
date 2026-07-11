import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

random_state = 42
target_col = "yield"
metric_label = "MAE"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def make_lgb_model():
    return lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="mae",
        random_state=random_state,
        verbose=-1,
    )


def make_cat_model():
    return CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=random_state,
        verbose=0,
    )


def drop_redundant_features(df):
    out = df.copy()
    cols_to_drop = [
        "id",
        "AverageOfUpperTRange",
        "AverageOfLowerTRange",
        "AverageRainingDays",
    ]
    existing = [c for c in cols_to_drop if c in out.columns]
    if existing:
        out = out.drop(columns=existing)
    return out


def score_variant(variant_name, build_predictors):
    data_train = skrub.var("data", train_part)
    lgb_pred_chain, cat_pred_chain = build_predictors(data_train)

    lgb_learner = lgb_pred_chain.skb.make_learner(fitted=True)
    cat_learner = cat_pred_chain.skb.make_learner(fitted=True)

    valid_pred_lgb = np.asarray(lgb_learner.predict({"data": valid_part}), dtype=float)
    valid_pred_cat = np.asarray(cat_learner.predict({"data": valid_part}), dtype=float)

    valid_pred = 0.5 * valid_pred_lgb + 0.5 * valid_pred_cat
    score = mean_absolute_error(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


def baseline_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    lgb_predictor = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(make_lgb_model(), y=y_train)
    cat_predictor = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(make_cat_model(), y=y_train)
    return lgb_predictor, cat_predictor


def no_catboost_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    lgb_predictor = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(make_lgb_model(), y=y_train)
    cat_predictor = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(make_lgb_model(), y=y_train)
    return lgb_predictor, cat_predictor


def drop_redundant_cols_graph(data_train):
    data_reduced = data_train.skb.apply_func(drop_redundant_features)
    X_train = data_reduced.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_reduced[target_col].skb.mark_as_y()

    lgb_predictor = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(make_lgb_model(), y=y_train)
    cat_predictor = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(make_cat_model(), y=y_train)
    return lgb_predictor, cat_predictor


def lgb_only_graph(data_train):
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    lgb_predictor = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(make_lgb_model(), y=y_train)
    cat_predictor = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(make_lgb_model(), y=y_train)
    return lgb_predictor, cat_predictor


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
scores["no_catboost_blend"] = score_variant("no_catboost_blend", no_catboost_graph)
scores["drop_redundant_cols"] = score_variant("drop_redundant_cols", drop_redundant_cols_graph)
scores["lgb_only"] = score_variant("lgb_only", lgb_only_graph)

best_variant = min(scores, key=scores.get)
best_score = scores[best_variant]
worst_variant = max(scores, key=scores.get)
worst_score = scores[worst_variant]
largest_effect = worst_score - scores["baseline"] if worst_variant != "baseline" else 0.0

print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")
print(f"Biggest contribution change vs baseline: {worst_variant} | delta_{metric_label}: {largest_effect}")