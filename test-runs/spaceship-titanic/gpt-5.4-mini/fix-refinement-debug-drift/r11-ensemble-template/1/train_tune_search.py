
import os
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

random_state = 42
n_iter = 8

input_dir = "./input"
train_path = os.path.join(input_dir, "train.csv")
test_path = os.path.join(input_dir, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = "Transported"

if train_df[target_col].dtype == bool:
    y_map = {False: 0, True: 1}
    inv_y_map = {0: False, 1: True}
else:
    y_map = {"False": 0, "True": 1, False: 0, True: 1}
    inv_y_map = {0: False, 1: True}

train_df = train_df.copy()
train_df[target_col] = train_df[target_col].map(y_map).astype(int)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state, stratify=train_df[target_col]
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def fe_func(df):
    out = df.copy()
    cabin_split = out["Cabin"].astype(str).str.split("/", expand=True)
    out["CabinDeck"] = cabin_split[0]
    out["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    out["CabinSide"] = cabin_split[2]
    out["Group"] = out["PassengerId"].astype(str).str.split("_", expand=True)[0]
    out["NameLen"] = out["Name"].astype(str).str.len()
    out["HasName"] = out["Name"].notna().astype(int)
    out["Spending"] = out[["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(0).sum(axis=1)
    out["NoSpending"] = (out["Spending"] == 0).astype(int)
    out["IsAdult"] = (out["Age"].fillna(out["Age"].median()) >= 18).astype(int)
    return out

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(fe_func)

X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

n_estimators = 250
lgbm_model = LGBMClassifier(
    n_estimators=n_estimators,
    learning_rate=skrub.choose_float(0.02, 0.2, log=True, default=0.05, name="learning_rate"),
    num_leaves=skrub.choose_int(16, 64, default=31, name="num_leaves"),
    max_depth=skrub.choose_int(3, 10, default=-1, name="max_depth"),
    min_child_samples=skrub.choose_int(5, 40, default=20, name="min_child_samples"),
    subsample=skrub.choose_float(0.7, 1.0, default=0.9, name="subsample"),
    colsample_bytree=skrub.choose_float(0.7, 1.0, default=0.9, name="colsample_bytree"),
    random_state=random_state,
    n_jobs=1,
    verbose=-1,
)

vectorizer = skrub.TableVectorizer()

pred = X_train.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_train)

search = pred.skb.make_randomized_search(
    n_iter=n_iter,
    n_jobs=1,
    random_state=random_state,
    fitted=True,
)

search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
if isinstance(valid_pred, (pd.DataFrame, pd.Series)):
    valid_pred = np.asarray(valid_pred).ravel()
else:
    valid_pred = np.asarray(valid_pred).ravel()

valid_pred_labels = (valid_pred > 0.5).astype(int)
final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred_labels)
print(f"Final Validation Performance: {final_validation_score}")

best_params = {}
grid = search.best_learner_.skb.describe_param_grid() if hasattr(search.best_learner_, "skb") else pred.skb.describe_param_grid()
try:
    bp = getattr(search, "best_params_", {})
    if bp:
        raw_values = list(bp.values())
        remaining = list(raw_values)
        tune_plan = {
            "tunable_params": [
                {"name": "learning_rate", "kind": "choose_float", "low": 0.02, "high": 0.2, "default": 0.05},
                {"name": "num_leaves", "kind": "choose_int", "low": 16, "high": 64, "default": 31},
                {"name": "max_depth", "kind": "choose_int", "low": 3, "high": 10, "default": -1},
                {"name": "min_child_samples", "kind": "choose_int", "low": 5, "high": 40, "default": 20},
                {"name": "subsample", "kind": "choose_float", "low": 0.7, "high": 1.0, "default": 0.9},
                {"name": "colsample_bytree", "kind": "choose_float", "low": 0.7, "high": 1.0, "default": 0.9},
            ]
        }
        for spec in tune_plan["tunable_params"]:
            name, kind = spec["name"], spec["kind"]
            if kind == "choose_int":
                match = next((v for v in remaining if spec["low"] <= int(round(float(v))) <= spec["high"]), spec.get("default"))
                best_params[name] = int(round(float(match)))
            elif kind == "choose_float":
                match = next((v for v in remaining if spec["low"] <= float(v) <= spec["high"]), spec.get("default"))
                best_params[name] = float(match)
            else:
                match = remaining.pop(0) if remaining else spec.get("default")
                best_params[name] = match
            if match in remaining:
                remaining.remove(match)
    else:
        best_params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        }
except Exception:
    best_params = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
