
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")

target_col = "Transported"
random_state = 42

train_df = pd.read_csv(TRAIN_PATH)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)


def engineer_passenger_features(df):
    out = df.copy()

    out = out.drop(columns=["PassengerId", "Name"], errors="ignore")

    if "Cabin" in out.columns:
        cabin_parts = out["Cabin"].astype("string").str.split("/", n=2, expand=True)
        if cabin_parts is not None:
            if 0 in cabin_parts.columns:
                out["CabinDeck"] = cabin_parts[0]
            if 1 in cabin_parts.columns:
                out["CabinNum"] = pd.to_numeric(cabin_parts[1], errors="coerce")
            if 2 in cabin_parts.columns:
                out["CabinSide"] = cabin_parts[2]
        out = out.drop(columns=["Cabin"], errors="ignore")

    return out


data_train_fe = data_train.skb.apply_func(engineer_passenger_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer_lgbm = skrub.TableVectorizer()
lgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=random_state,
    verbose=-1,
)
predictor_lgbm = X_train.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_train)
learner_lgbm = predictor_lgbm.skb.make_learner(fitted=True)

vectorizer_cat = skrub.TableVectorizer()
cat_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    random_state=random_state,
    verbose=0,
)
predictor_cat = X_train.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_train)
learner_cat = predictor_cat.skb.make_learner(fitted=True)


def extract_positive_score(learner, df):
    try:
        proba = np.asarray(learner.predict_proba({"data": df}))
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1].astype(float)
        return proba.ravel().astype(float)
    except Exception:
        pred = np.asarray(learner.predict({"data": df})).ravel()
        pred_series = pd.Series(pred).astype(str).map({"True": 1.0, "False": 0.0})
        if pred_series.notna().all():
            return pred_series.to_numpy(dtype=float)
        return pd.Series(pred).astype(float).to_numpy()


valid_true = pd.Series(valid_part[target_col], index=valid_part.index).astype(str).map(
    {"True": True, "False": False}
).fillna(valid_part[target_col]).astype(bool)
valid_true_num = valid_true.astype(int).to_numpy()

p_lgbm = extract_positive_score(learner_lgbm, valid_part)
p_cat = extract_positive_score(learner_cat, valid_part)

lgbm_pred = p_lgbm >= 0.5
cat_pred = p_cat >= 0.5
avg_pred = ((p_lgbm + p_cat) / 2.0) >= 0.5

conf_choice_scores = np.where(
    np.abs(p_lgbm - 0.5) >= np.abs(p_cat - 0.5),
    p_lgbm,
    p_cat,
)
conf_pred = conf_choice_scores >= 0.5

agree_mask = lgbm_pred == cat_pred
gated_pred = np.where(agree_mask, lgbm_pred, conf_pred)

meta_X_valid = np.column_stack([p_lgbm, p_cat])
stacker = LogisticRegression(class_weight="balanced", random_state=random_state, max_iter=1000)
stacker.fit(meta_X_valid, valid_true_num)
stack_pred = stacker.predict(meta_X_valid).astype(bool)

candidate_scores = {
    "lgbm": accuracy_score(valid_true, lgbm_pred),
    "cat": accuracy_score(valid_true, cat_pred),
    "avg": accuracy_score(valid_true, avg_pred),
    "conf": accuracy_score(valid_true, conf_pred),
    "gated": accuracy_score(valid_true, gated_pred),
    "stack": accuracy_score(valid_true, stack_pred),
}

best_rule = max(candidate_scores, key=candidate_scores.get)
final_validation_score = candidate_scores[best_rule]

print(f"Best merge rule: {best_rule}")
print(f"Final Validation Performance: {final_validation_score}")

TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
test_df = pd.read_csv(TEST_PATH)

data_full = skrub.var("data", train_df)
data_full_fe = data_full.skb.apply_func(engineer_passenger_features)
X_full = data_full_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full_fe[target_col].skb.mark_as_y()

full_predictor_lgbm = X_full.skb.apply(vectorizer_lgbm).skb.apply(lgbm_model, y=y_full)
full_learner_lgbm = full_predictor_lgbm.skb.make_learner(fitted=True)

full_predictor_cat = X_full.skb.apply(vectorizer_cat).skb.apply(cat_model, y=y_full)
full_learner_cat = full_predictor_cat.skb.make_learner(fitted=True)

test_p_lgbm = extract_positive_score(full_learner_lgbm, test_df)
test_p_cat = extract_positive_score(full_learner_cat, test_df)

test_lgbm_pred = test_p_lgbm >= 0.5
test_cat_pred = test_p_cat >= 0.5
test_avg_pred = ((test_p_lgbm + test_p_cat) / 2.0) >= 0.5

test_conf_choice_scores = np.where(
    np.abs(test_p_lgbm - 0.5) >= np.abs(test_p_cat - 0.5),
    test_p_lgbm,
    test_p_cat,
)
test_conf_pred = test_conf_choice_scores >= 0.5

test_agree_mask = test_lgbm_pred == test_cat_pred
test_gated_pred = np.where(test_agree_mask, test_lgbm_pred, test_conf_pred)

meta_X_full = np.column_stack([p_lgbm, p_cat])
full_stacker = LogisticRegression(
    class_weight="balanced",
    random_state=random_state,
    max_iter=1000,
)
full_stacker.fit(meta_X_full, valid_true_num)

meta_X_test = np.column_stack([test_p_lgbm, test_p_cat])
test_stack_pred = full_stacker.predict(meta_X_test).astype(bool)

test_rule_predictions = {
    "lgbm": test_lgbm_pred,
    "cat": test_cat_pred,
    "avg": test_avg_pred,
    "conf": test_conf_pred,
    "gated": test_gated_pred,
    "stack": test_stack_pred,
}

test_pred = pd.Series(test_rule_predictions[best_rule]).astype(bool)

os.makedirs("./final", exist_ok=True)
submission = pd.DataFrame(
    {
        "PassengerId": test_df["PassengerId"],
        "Transported": test_pred,
    }
)
submission.to_csv("./final/submission.csv", index=False)
