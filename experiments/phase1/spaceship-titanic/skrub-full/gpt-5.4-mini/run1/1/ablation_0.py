import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

train_df = pd.read_csv("./input/train.csv")
target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def add_features(df):
    df = df.copy()

    cabin = df["Cabin"].fillna("")
    cabin_split = cabin.str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0].replace("", np.nan)
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2].replace("", np.nan)

    name = df["Name"].fillna("")
    df["Surname"] = name.str.split(" ").str[-1].replace("", np.nan)

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpent"] = df[spend_cols].fillna(0).sum(axis=1)
    df["NoSpend"] = (df["TotalSpent"] == 0).astype(int)

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[-1, 12, 18, 30, 50, 120],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    )
    return df

def run_variant(variant_name, use_features=True, use_cat=True, use_lgb=True):
    data_train = skrub.var("data", train_part)
    if use_features:
        data_train = data_train.skb.apply_func(add_features)

    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    preds = []

    if use_cat:
        cat_model = CatBoostClassifier(
            verbose=0,
            loss_function="Logloss",
            random_seed=42,
        )
        cat_pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(cat_model, y=y_train)
        cat_learner = cat_pred.skb.make_learner(fitted=True)
        cat_valid_pred = np.asarray(cat_learner.predict({"data": valid_part})).ravel()
        cat_valid_pred = cat_valid_pred >= 0.5 if cat_valid_pred.dtype != bool else cat_valid_pred
        preds.append(cat_valid_pred.astype(float))

    if use_lgb:
        lgb_model = LGBMClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            verbose=-1,
        )
        lgb_pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(lgb_model, y=y_train)
        lgb_learner = lgb_pred.skb.make_learner(fitted=True)
        lgb_valid_pred = np.asarray(lgb_learner.predict({"data": valid_part})).ravel()
        lgb_valid_pred = lgb_valid_pred >= 0.5 if lgb_valid_pred.dtype != bool else lgb_valid_pred
        preds.append(lgb_valid_pred.astype(float))

    if len(preds) == 1:
        ensemble_valid_pred = preds[0] >= 0.5
    else:
        ensemble_valid_pred = (sum(preds) / len(preds)) >= 0.5

    score = accuracy_score(valid_part[target_col], ensemble_valid_pred)
    print(f"Ablation[{variant_name}] accuracy: {score}")
    return score

baseline_score = run_variant("baseline", use_features=True, use_cat=True, use_lgb=True)
no_feature_score = run_variant("no_feature_engineering", use_features=False, use_cat=True, use_lgb=True)
cat_only_score = run_variant("catboost_only", use_features=True, use_cat=True, use_lgb=False)

scores = {
    "baseline": baseline_score,
    "no_feature_engineering": no_feature_score,
    "catboost_only": cat_only_score,
}
best_variant = max(scores, key=scores.get)
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | accuracy: {best_score}")
print(f"Final Validation Performance: {baseline_score}")