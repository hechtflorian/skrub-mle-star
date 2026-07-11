import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

random_state = 42
np.random.seed(random_state)
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

# Keep preprocessing intentionally close to the original solution for the baseline.
def preprocess_df(df, use_name=True, use_planet_deck=True, use_age_group=True, drop_passenger_id=False):
    df = df.copy()

    if "Cabin" in df.columns:
        cabin = df["Cabin"].astype("string")
        cabin_split = cabin.str.split("/", expand=True)
        if cabin_split.shape[1] >= 3:
            df["CabinDeck"] = cabin_split[0]
            df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
            df["CabinSide"] = cabin_split[2]
        else:
            df["CabinDeck"] = pd.NA
            df["CabinNum"] = np.nan
            df["CabinSide"] = pd.NA
        df = df.drop(columns=["Cabin"], errors="ignore")

    if use_name and "Name" in df.columns:
        df["Surname"] = df["Name"].astype("string").str.split().str[-1]
        df = df.drop(columns=["Name"], errors="ignore")

    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype).startswith("string"):
            df[col] = df[col].fillna("missing")
        else:
            df[col] = df[col].fillna(df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 0)

    if use_planet_deck and "HomePlanet" in df.columns and "CabinDeck" in df.columns:
        df["PlanetDeck"] = df["HomePlanet"].astype("string") + "_" + df["CabinDeck"].astype("string")

    if use_age_group and "Age" in df.columns:
        df["AgeGroup"] = pd.cut(
            df["Age"],
            bins=[-np.inf, 12, 18, 30, 50, np.inf],
            labels=["child", "teen", "young", "adult", "senior"],
        ).astype("string")

    if drop_passenger_id:
        df = df.drop(columns=["PassengerId"], errors="ignore")

    return df


def score_variant(variant_name, use_name=True, use_planet_deck=True, use_age_group=True, drop_passenger_id=False,
                  drop_cabin_num=False, drop_surname=False, use_single_model=False):
    train_df_proc = preprocess_df(
        train_df,
        use_name=use_name,
        use_planet_deck=use_planet_deck,
        use_age_group=use_age_group,
    )

    if drop_cabin_num:
        train_df_proc = train_df_proc.drop(columns=["CabinNum"], errors="ignore")
    if drop_surname:
        train_df_proc = train_df_proc.drop(columns=["Surname"], errors="ignore")
    if drop_passenger_id:
        train_df_proc = train_df_proc.drop(columns=["PassengerId"], errors="ignore")

    train_idx, valid_idx = train_test_split(
        np.arange(len(train_df_proc)),
        test_size=0.2,
        random_state=random_state,
        stratify=train_df_proc[target_col].astype(int),
    )

    train_part = train_df_proc.iloc[train_idx].copy()
    valid_part = train_df_proc.iloc[valid_idx].copy()

    data_train = skrub.var("data", train_part)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    if use_single_model:
        vectorizer = skrub.TableVectorizer()
        cat_model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=0,
        )
        pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
        learner = pred.skb.make_learner(fitted=True)
        valid_pred = np.asarray(learner.predict({"data": valid_part})).reshape(-1).astype(float)
        valid_label = valid_pred >= 0.5
    else:
        vectorizer = skrub.TableVectorizer()
        cat_model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_seed=random_state,
            verbose=0,
        )
        lgbm_model = LGBMClassifier(
            random_state=random_state,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
        )

        cat_pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
        lgbm_pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(lgbm_model, y=y_train)

        cat_learner = cat_pred.skb.make_learner(fitted=True)
        lgbm_learner = lgbm_pred.skb.make_learner(fitted=True)

        valid_cat = np.asarray(cat_learner.predict({"data": valid_part})).reshape(-1).astype(float)
        valid_lgbm = np.asarray(lgbm_learner.predict({"data": valid_part})).reshape(-1).astype(float)

        valid_blend = 0.55 * valid_cat + 0.45 * valid_lgbm
        valid_label = valid_blend >= 0.5

    score = accuracy_score(valid_part[target_col].astype(bool), valid_label)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score


scores = {}

# Baseline: same estimator classes and same validation split logic as the original solution.
scores["baseline"] = score_variant(
    "baseline",
    use_name=True,
    use_planet_deck=True,
    use_age_group=True,
    use_single_model=False,
)

# Ablation 1: disable the Name-derived surname feature.
scores["no_name_feature"] = score_variant(
    "no_name_feature",
    use_name=False,
    use_planet_deck=True,
    use_age_group=True,
    use_single_model=False,
)

# Ablation 2: disable the AgeGroup derived feature.
scores["no_age_group"] = score_variant(
    "no_age_group",
    use_name=True,
    use_planet_deck=True,
    use_age_group=False,
    use_single_model=False,
)

# Ablation 3: remove the high-cardinality PassengerId column.
scores["drop_passenger_id"] = score_variant(
    "drop_passenger_id",
    use_name=True,
    use_planet_deck=True,
    use_age_group=True,
    drop_passenger_id=True,
    use_single_model=False,
)

# Ablation 4: single-model baseline without the LightGBM blend.
scores["catboost_only"] = score_variant(
    "catboost_only",
    use_name=True,
    use_planet_deck=True,
    use_age_group=True,
    use_single_model=True,
)

best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {best_score}")