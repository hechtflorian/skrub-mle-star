
import os
import json
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
train_df = pd.read_csv(train_path)

tune_plan = {
    "tunable_params": [
        {
            "name": "learning_rate",
            "kind": "choose_float",
            "low": 0.015,
            "high": 0.04,
            "default": 0.025,
        },
        {
            "name": "num_leaves",
            "kind": "choose_int",
            "low": 24,
            "high": 40,
            "default": 31,
        },
    ]
}

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
    n_estimators=300,
    learning_rate=skrub.choose_float(low=0.015, high=0.04, default=0.025, name="learning_rate"),
    num_leaves=skrub.choose_int(24, 40, default=31, name="num_leaves"),
    max_depth=-1,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=random_state,
    n_jobs=-1,
    verbose=-1,
)

lgbm_pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(lgbm_model, y=y_train)
lgbm_learner = lgbm_pred.skb.make_learner(fitted=True)

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

search = lgbm_pred.skb.make_randomized_search(n_iter=5, n_jobs=1, random_state=random_state, fitted=True)
search.fit({"data": train_part})

valid_pred_lgbm = search.best_learner_.predict({"data": valid_part})
valid_pred_lgbm = np.asarray(valid_pred_lgbm).ravel()

valid_pred_rf = rf_model.predict_proba(meta_valid_X)[:, 1]
valid_pred_lr = lr_model.predict_proba(meta_valid_X)[:, 1]

valid_pred_ens = (0.6 * valid_pred_lgbm + 0.2 * valid_pred_rf + 0.2 * valid_pred_lr) >= 0.5
final_validation_score = accuracy_score(valid_part[target_col].astype(bool), valid_pred_ens.astype(bool))
print(f"Final Validation Performance: {final_validation_score}")

raw_best = list(search.best_params_.values())
remaining = list(raw_best)
best_params = {}
for spec in tune_plan["tunable_params"]:
    if spec["kind"] == "choose_int":
        match = next((v for v in remaining if spec["low"] <= int(round(float(v))) <= spec["high"]), spec["default"])
        best_params[spec["name"]] = int(round(float(match)))
        if match in remaining:
            remaining.remove(match)
    elif spec["kind"] == "choose_float":
        match = next((v for v in remaining if spec["low"] <= float(v) <= spec["high"]), spec["default"])
        best_params[spec["name"]] = float(match)
        if match in remaining:
            remaining.remove(match)
    else:
        match = remaining.pop(0) if remaining else spec["default"]
        best_params[spec["name"]] = match

print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
