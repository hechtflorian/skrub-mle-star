
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

warnings.filterwarnings("ignore")

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

def add_features(df):
    out = df.copy()

    cabin = out["Cabin"].fillna("U/U/U").astype(str).str.split("/", expand=True)
    out["Deck"] = cabin[0]
    out["Num"] = pd.to_numeric(cabin[1], errors="coerce")
    out["Side"] = cabin[2]

    for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["TotalSpend"] = out[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["Spent"] = (out["TotalSpend"] > 0).astype(float)
    out["Age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["AgeBucket"] = pd.cut(
        out["Age"],
        bins=[-1, 12, 18, 25, 35, 50, 80, 200],
        labels=False,
        include_lowest=True,
    )
    out["CryoSleep"] = out["CryoSleep"].astype("object")
    out["VIP"] = out["VIP"].astype("object")
    out["HasCabin"] = out["Cabin"].notna().astype(float)

    return out.drop(columns=["Cabin", "Name"], errors="ignore")

def build_sklearn_meta_features(df):
    feat = add_features(df)
    X = feat.drop(columns=[target_col], errors="ignore")
    num_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )
    X_arr = pre.fit_transform(X)
    return X_arr, pre

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state, stratify=train_df[target_col].astype(int)
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

lgbm_model = LGBMClassifier(
    n_estimators=600,
    learning_rate=0.025,
    num_leaves=31,
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

lgbm_pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(lgbm_model, y=y_train)
lgbm_learner = lgbm_pred.skb.make_learner(fitted=True)
valid_pred_lgbm = lgbm_learner.predict({"data": valid_part})
valid_pred_lgbm = np.asarray(valid_pred_lgbm).ravel()

train_part_fe = add_features(train_part)
valid_part_fe = add_features(valid_part)

meta_train_X, meta_pre = build_sklearn_meta_features(train_part)
meta_valid_X = meta_pre.transform(valid_part_fe.drop(columns=[target_col], errors="ignore"))

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=random_state,
    n_jobs=-1,
)

lr_model = LogisticRegression(
    max_iter=2000,
    C=1.0,
    solver="liblinear",
    random_state=random_state,
)

rf_model.fit(meta_train_X, train_part[target_col].astype(int).values)
lr_model.fit(meta_train_X, train_part[target_col].astype(int).values)

valid_pred_rf = rf_model.predict_proba(meta_valid_X)[:, 1]
valid_pred_lr = lr_model.predict_proba(meta_valid_X)[:, 1]

valid_pred_ens = (0.6 * valid_pred_lgbm + 0.2 * valid_pred_rf + 0.2 * valid_pred_lr) >= 0.5
final_validation_score = accuracy_score(valid_part[target_col].astype(bool), valid_pred_ens.astype(bool))
print(f"Final Validation Performance: {final_validation_score}")

data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(add_features)
X_full = data_full_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].skb.mark_as_y()

full_pred = X_full.skb.apply(skrub.TableVectorizer()).skb.apply(lgbm_model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)
test_pred_lgbm = full_learner.predict({"data": test_df})
test_pred_lgbm = np.asarray(test_pred_lgbm).ravel()

full_meta_X, full_meta_pre = build_sklearn_meta_features(train_df)
test_meta_X = full_meta_pre.transform(add_features(test_df).drop(columns=[target_col], errors="ignore"))

rf_model_full = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    random_state=random_state,
    n_jobs=-1,
)
lr_model_full = LogisticRegression(
    max_iter=2000,
    C=1.0,
    solver="liblinear",
    random_state=random_state,
)
rf_model_full.fit(full_meta_X, train_df[target_col].astype(int).values)
lr_model_full.fit(full_meta_X, train_df[target_col].astype(int).values)

test_pred_rf = rf_model_full.predict_proba(test_meta_X)[:, 1]
test_pred_lr = lr_model_full.predict_proba(test_meta_X)[:, 1]

test_pred_ens = (0.6 * test_pred_lgbm + 0.2 * test_pred_rf + 0.2 * test_pred_lr) >= 0.5

submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Transported": test_pred_ens.astype(bool)})
submission.to_csv("submission.csv", index=False)
