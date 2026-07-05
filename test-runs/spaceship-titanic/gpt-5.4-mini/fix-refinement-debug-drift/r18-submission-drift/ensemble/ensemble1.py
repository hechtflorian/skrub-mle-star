
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from lightgbm import LGBMClassifier

random_state = 42
target_col = "Transported"

train_df = pd.read_csv("./input/train.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def add_features(df):
    df = df.copy()
    cabin = df["Cabin"].astype("string")
    cabin_parts = cabin.str.split("/", expand=True)
    df["CabinDeck"] = cabin_parts[0]
    df["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
    df["CabinSide"] = cabin_parts[2]

    name = df["Name"].astype("string")
    df["Surname"] = name.str.split(" ", n=1, expand=True)[1]
    df["FirstName"] = name.str.split(" ", n=1, expand=True)[0]

    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["SpendingMean"] = df[spend_cols].mean(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)

    df["FamilySize"] = df.groupby("Surname")["Surname"].transform("size")
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 25, 35, 50, 80],
        labels=["child", "teen", "young_adult", "adult", "mid_age", "senior"],
        include_lowest=True,
    )
    return df

def extract_scores(learner, df):
    try:
        proba = np.asarray(learner.predict_proba({"data": df}))
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(float).ravel()
        return proba.astype(float).ravel()
    except Exception:
        pred = np.asarray(learner.predict({"data": df}))
        if pred.dtype == bool:
            return pred.astype(float).ravel()
        return pred.astype(float).ravel()

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_features)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_1 = skrub.TableVectorizer()
vectorizer_2 = skrub.TableVectorizer()

model_1 = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

model_2 = LGBMClassifier(
    n_estimators=450,
    learning_rate=0.03,
    num_leaves=63,
    random_state=random_state + 1,
    n_jobs=-1,
    verbose=-1,
)

pred_1 = X_train.skb.apply(vectorizer_1).skb.apply(
    SimpleImputer(strategy="median")
).skb.apply(model_1, y=y_train)

pred_2 = X_train.skb.apply(vectorizer_2).skb.apply(
    SimpleImputer(strategy="median")
).skb.apply(model_2, y=y_train)

learner_1 = pred_1.skb.make_learner(fitted=True)
learner_2 = pred_2.skb.make_learner(fitted=True)

valid_score_1 = extract_scores(learner_1, valid_part)
valid_score_2 = extract_scores(learner_2, valid_part)

dist_1 = np.abs(valid_score_1 - 0.5)
dist_2 = np.abs(valid_score_2 - 0.5)

same_side = ((valid_score_1 - 0.5) * (valid_score_2 - 0.5)) >= 0
both_close = (dist_1 < 0.05) & (dist_2 < 0.05)

best_is_1 = dist_1 >= dist_2
conservative_avg = np.where(
    best_is_1,
    0.85 * valid_score_1 + 0.15 * valid_score_2,
    0.85 * valid_score_2 + 0.15 * valid_score_1,
)

agree_blend = conservative_avg
disagree_blend = np.where(best_is_1, valid_score_1, valid_score_2)

final_score = np.where(same_side, agree_blend, disagree_blend)
final_score = np.where(both_close, valid_score_1, final_score)

valid_pred = final_score >= 0.5
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)

print(f"Final Validation Performance: {final_validation_score}")
