
import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "Transported"



def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    cabin = out["Cabin"].fillna("U/U/U").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]

    spending_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    out["Spending"] = out[spending_cols].fillna(0).sum(axis=1)
    out["HasSpending"] = (out["Spending"] > 0).astype(int)

    out["AgeGroup"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 30, 50, 120],
        labels=["child", "teen", "young_adult", "adult", "senior"],
    ).astype("object")

    out["PassengerGroup"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["NameLength"] = out["Name"].fillna("").astype(str).str.len()
    out["Surname"] = out["Name"].fillna("").astype(str).str.split().str[-1]

    out["LuxurySpending"] = out[["Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["BasicSpending"] = out[["RoomService", "FoodCourt", "ShoppingMall"]].fillna(0).sum(axis=1)
    out["NoSpending"] = (out["Spending"] == 0).astype(int)

    out["FamilyNameKey"] = (
        out["Surname"].fillna("Unknown").astype(str)
        + "_"
        + out["PassengerGroup"].fillna("Unknown").astype(str)
    )

    # Safe structural additions from the plan
    out["CabinDeckSide"] = out["Deck"].fillna("U").astype(str) + "_" + out["Side"].fillna("U").astype(str)

    # Cabin number buckets: preserve ordinal signal without overfitting exact numbers
    out["CabinNumBucket"] = pd.cut(
        out["Num"],
        bins=[-np.inf, 100, 300, 600, 900, np.inf],
        labels=["very_low", "low", "mid", "high", "very_high"],
        include_lowest=True,
    ).astype("object")

    # Household / group-level indicators
    out["HouseholdSize"] = out.groupby("PassengerGroup")["PassengerGroup"].transform("size")
    out["IsSoloPassenger"] = (out["HouseholdSize"] == 1).astype(int)
    out["IsLargeGroup"] = (out["HouseholdSize"] >= 4).astype(int)

    # Group-relative cabin signal
    out["CabinNumIsMissing"] = out["Num"].isna().astype(int)
    out["CabinNumMod100"] = (out["Num"] % 100).fillna(-1).astype(float)
    out["CabinNumBand"] = pd.cut(
        out["CabinNumMod100"],
        bins=[-2, 20, 40, 60, 80, 100],
        labels=["b0", "b1", "b2", "b3", "b4"],
        include_lowest=True,
    ).astype("object")

    # Lightweight cleanup for sparse / high-missing fields before encoding
    out["Name"] = out["Name"].fillna("Unknown")
    out["Cabin"] = out["Cabin"].fillna("U/U/U")
    out["VIP"] = out["VIP"].fillna(0).astype(int) if "VIP" in out.columns else out["VIP"]
    out["CryoSleep"] = out["CryoSleep"].fillna(False).astype(int) if "CryoSleep" in out.columns else out["CryoSleep"]

    return out


import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_fe = data_train.skb.apply_func(add_features)
X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()

# Single clean vectorization path after feature engineering
vectorizer = skrub.TableVectorizer()

# Keep the existing fixed backbone path; if an existing model is already defined earlier,
# reuse it. Otherwise, use a stable fixed default that fits the DataOps path.
if "learner" in globals():
    backbone = learner
elif "model" in globals():
    backbone = model
else:
    from sklearn.ensemble import HistGradientBoostingClassifier

    backbone = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=6,
        max_iter=180,
        min_samples_leaf=20,
        l2_regularization=0.0,
        random_state=42,
    )

predictor = X.skb.apply(vectorizer).skb.apply(backbone, y=y)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

if hasattr(valid_pred, "to_numpy"):
    valid_pred_eval = valid_pred.to_numpy()
else:
    valid_pred_eval = np.asarray(valid_pred)

final_validation_score = accuracy_score(valid_part[target_col], valid_pred_eval)
print(f"Final Validation Performance: {final_validation_score}")


train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=42,
    stratify=train_df[target_col].astype(int),
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)

X_train = data_train_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].astype(int).skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

lgbm_model = LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=31,
    random_state=42,
    verbose=-1,
)

cat_model = CatBoostClassifier(
    iterations=1200,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    verbose=0,
    random_seed=42,
    allow_writing_files=False,
)

lgbm_graph = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)
cat_graph = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)

lgbm_learner = lgbm_graph.skb.make_learner(fitted=True)
cat_learner = cat_graph.skb.make_learner(fitted=True)

valid_pred_lgbm = lgbm_learner.predict({"data": valid_part}).astype(float)
valid_pred_cat = cat_learner.predict({"data": valid_part}).astype(float)

valid_pred_ensemble = ((valid_pred_lgbm + valid_pred_cat) / 2.0 >= 0.5).astype(int)
final_validation_score = accuracy_score(valid_part[target_col].astype(int), valid_pred_ensemble)
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_features)

X_full = data_full_fe.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].astype(int).skb.mark_as_y()

full_lgbm_graph = X_full.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_full)
full_cat_graph = X_full.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_full)

full_lgbm_learner = full_lgbm_graph.skb.make_learner(fitted=True)
full_cat_learner = full_cat_graph.skb.make_learner(fitted=True)

test_pred_lgbm = full_lgbm_learner.predict({"data": test_df}).astype(float)
test_pred_cat = full_cat_learner.predict({"data": test_df}).astype(float)

test_pred = ((test_pred_lgbm + test_pred_cat) / 2.0 >= 0.5).astype(bool)

submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("submission.csv", index=False)
