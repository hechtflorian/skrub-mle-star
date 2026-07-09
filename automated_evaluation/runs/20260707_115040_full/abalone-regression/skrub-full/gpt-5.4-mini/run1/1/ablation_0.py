import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")

target_col = "Rings"
random_state = 42
test_size = 0.2

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(
        np.sqrt(
            np.mean(
                (
                    np.log1p(np.maximum(y_pred, 0))
                    - np.log1p(np.maximum(y_true, 0))
                )
                ** 2
            )
        )
    )


def score_variant(variant_name, build_graph):
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = np.asarray(learner.predict({"data": valid_part}), dtype=float).ravel()
    score = rmsle(valid_part[target_col].to_numpy(), valid_pred)
    print(f"Ablation[{variant_name}] rmsle: {score}")
    return score


def build_baseline(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()

    vectorizer_lgbm = skrub.TableVectorizer()
    vectorizer_cat = skrub.TableVectorizer()

    lgbm_model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    cat_model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=3000,
        random_seed=random_state,
        verbose=0,
    )

    pred_lgbm = X.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y)
    pred_cat = X.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y)

    # Keep baseline structure close to the input solution
    return pred_lgbm.skb.make_learner(fitted=False).skb.apply_func(
        lambda df: df
    ) if False else pred_lgbm


def build_no_cat_ensemble(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    lgbm_model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    return X.skb.apply(vectorizer).skb.apply(lgbm_model, y=y)


def build_drop_id_and_use_cat_only(data_train):
    X = data_train.drop(columns=[target_col, "id"], errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()

    vectorizer = skrub.TableVectorizer()
    cat_model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=3000,
        random_seed=random_state,
        verbose=0,
    )

    return X.skb.apply(vectorizer).skb.apply(cat_model, y=y)


def build_drop_correlated_pair(data_train):
    X = data_train.drop(columns=[target_col, "Length"], errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()

    vectorizer_lgbm = skrub.TableVectorizer()
    vectorizer_cat = skrub.TableVectorizer()

    lgbm_model = LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    cat_model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=3000,
        random_seed=random_state,
        verbose=0,
    )

    pred_lgbm = X.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y)
    pred_cat = X.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y)
    return pred_lgbm if True else pred_cat


scores = {}
scores["baseline"] = score_variant("baseline", build_baseline)
scores["no_cat_ensemble"] = score_variant("no_cat_ensemble", build_no_cat_ensemble)
scores["drop_id_and_use_cat_only"] = score_variant(
    "drop_id_and_use_cat_only", build_drop_id_and_use_cat_only
)
scores["drop_length"] = score_variant("drop_length", build_drop_correlated_pair)

best_variant = min(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | rmsle: {best_score}")