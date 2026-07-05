
import os
import numpy as np
import pandas as pd
import skrub
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

def add_features(df):
    out = df.copy()
    out["CabinDeck"] = out["Cabin"].fillna("X").astype(str).str.split("/").str[0]
    out["CabinNum"] = pd.to_numeric(out["Cabin"].fillna("X").astype(str).str.split("/").str[1], errors="coerce")
    out["CabinSide"] = out["Cabin"].fillna("X").astype(str).str.split("/").str[2]
    out["Group"] = out["PassengerId"].astype(str).str.split("_").str[0]
    out["GroupSize"] = out["Group"].map(out.groupby("Group").size())
    out["Surname"] = out["Name"].fillna("Unknown").astype(str).str.split(" ").str[-1]
    out["TotalSpend"] = out[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["NoSpend"] = (out["TotalSpend"] == 0).astype(int)
    out["IsAlone"] = (out["GroupSize"] == 1).astype(int)
    out["AgeGroup"] = pd.cut(out["Age"], bins=[-1, 12, 18, 25, 35, 50, 65, 200], labels=False)
    out["SpendPerAge"] = (out["TotalSpend"] / out["Age"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["CabinKnown"] = (out["Cabin"].notna()).astype(int)
    return out

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train = data_train.skb.apply_func(add_features)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
vectorizer_cat = skrub.TableVectorizer()

model_lgbm = lgb.LGBMClassifier(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

model_cat = CatBoostClassifier(
    iterations=300,
    depth=6,
    learning_rate=0.08,
    loss_function="Logloss",
    random_seed=random_state,
    verbose=0,
)

pred_lgbm = X_train.skb.apply(vectorizer_lgbm).skb.apply(model_lgbm, y=y_train)
pred_cat = X_train.skb.apply(vectorizer_cat).skb.apply(model_cat, y=y_train)

learner_lgbm = pred_lgbm.skb.make_learner(fitted=True)
learner_cat = pred_cat.skb.make_learner(fitted=True)

valid_features = add_features(valid_part.copy())

valid_pred_lgbm = learner_lgbm.predict({"data": valid_features})
valid_pred_cat = learner_cat.predict({"data": valid_features})

valid_pred_lgbm = np.asarray(valid_pred_lgbm)
valid_pred_cat = np.asarray(valid_pred_cat)

if valid_pred_lgbm.dtype != bool:
    if np.issubdtype(valid_pred_lgbm.dtype, np.floating):
        valid_pred_lgbm = valid_pred_lgbm >= 0.5
    else:
        valid_pred_lgbm = pd.Series(valid_pred_lgbm).astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()

if valid_pred_cat.dtype != bool:
    if np.issubdtype(valid_pred_cat.dtype, np.floating):
        valid_pred_cat = valid_pred_cat >= 0.5
    else:
        valid_pred_cat = pd.Series(valid_pred_cat).astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()

blend_pred = (0.6 * valid_pred_lgbm.astype(float) + 0.4 * valid_pred_cat.astype(float)) >= 0.5

final_validation_score = accuracy_score(valid_part[target_col].astype(int), blend_pred.astype(int))
print(f"Final Validation Performance: {final_validation_score}")
