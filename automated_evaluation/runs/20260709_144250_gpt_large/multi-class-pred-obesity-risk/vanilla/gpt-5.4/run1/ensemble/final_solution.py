
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

TARGET = "NObeyesdad"
ID = "id"

train_path = "./input/train.csv"
test_path = "./input/test.csv"

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

cat_cols = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]

X = train.drop(columns=[ID, TARGET])
y = train[TARGET]
X_test = test.drop(columns=[ID])

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

base_model_params = {
    "loss_function": "MultiClass",
    "eval_metric": "Accuracy",
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "random_seed": 42,
    "verbose": False,
    "thread_count": -1
}

full_model = CatBoostClassifier(**base_model_params)
full_model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_valid, y_valid),
    use_best_model=True,
    early_stopping_rounds=50
)

class_names = list(full_model.classes_)

full_valid_proba = pd.DataFrame(
    full_model.predict_proba(X_valid),
    columns=class_names,
    index=X_valid.index
)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

cv_valid_proba = pd.DataFrame(
    np.zeros((len(X_valid), len(class_names))),
    columns=class_names,
    index=X_valid.index
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_tr = X_train.iloc[tr_idx]
    y_tr = y_train.iloc[tr_idx]
    X_va = X_train.iloc[va_idx]
    y_va = y_train.iloc[va_idx]

    fold_model = CatBoostClassifier(**base_model_params)
    fold_model.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_va, y_va),
        use_best_model=True,
        early_stopping_rounds=50
    )

    fold_classes = list(fold_model.classes_)

    fold_valid_on_holdout = pd.DataFrame(
        fold_model.predict_proba(X_valid),
        columns=fold_classes,
        index=X_valid.index
    ).reindex(columns=class_names, fill_value=0.0)

    cv_valid_proba += fold_valid_on_holdout / skf.n_splits

def row_confidence_max(proba_df):
    arr = proba_df.values
    return arr.max(axis=1)

def row_confidence_margin(proba_df):
    arr = proba_df.values
    sorted_arr = np.sort(arr, axis=1)
    return sorted_arr[:, -1] - sorted_arr[:, -2]

def predict_from_proba(proba_df):
    return proba_df.idxmax(axis=1).values

def evaluate_merge_rule(full_valid_proba, cv_valid_proba, y_valid, rule_name, class_names):
    if rule_name == "fixed_05_05":
        merged = 0.5 * full_valid_proba + 0.5 * cv_valid_proba
    elif rule_name == "fixed_04_06_cv":
        merged = 0.4 * full_valid_proba + 0.6 * cv_valid_proba
    elif rule_name == "dynamic_max":
        conf_full = row_confidence_max(full_valid_proba)
        conf_cv = row_confidence_max(cv_valid_proba)
        w_full = conf_full / (conf_full + conf_cv + 1e-12)
        w_full = 0.25 + 0.5 * w_full
        w_cv = 1.0 - w_full
        merged = pd.DataFrame(
            w_full[:, None] * full_valid_proba.values + w_cv[:, None] * cv_valid_proba.values,
            columns=class_names,
            index=full_valid_proba.index
        )
    elif rule_name == "dynamic_margin":
        conf_full = row_confidence_margin(full_valid_proba)
        conf_cv = row_confidence_margin(cv_valid_proba)
        w_full = conf_full / (conf_full + conf_cv + 1e-12)
        w_full = 0.25 + 0.5 * w_full
        w_cv = 1.0 - w_full
        merged = pd.DataFrame(
            w_full[:, None] * full_valid_proba.values + w_cv[:, None] * cv_valid_proba.values,
            columns=class_names,
            index=full_valid_proba.index
        )
    elif rule_name == "gated_max":
        conf_full = row_confidence_max(full_valid_proba)
        conf_cv = row_confidence_max(cv_valid_proba)
        diff = conf_full - conf_cv
        threshold = 0.05
        w_full = np.where(diff > threshold, 0.7, np.where(diff < -threshold, 0.3, 0.5))
        w_cv = 1.0 - w_full
        merged = pd.DataFrame(
            w_full[:, None] * full_valid_proba.values + w_cv[:, None] * cv_valid_proba.values,
            columns=class_names,
            index=full_valid_proba.index
        )
    elif rule_name == "gated_margin":
        conf_full = row_confidence_margin(full_valid_proba)
        conf_cv = row_confidence_margin(cv_valid_proba)
        diff = conf_full - conf_cv
        threshold = 0.03
        w_full = np.where(diff > threshold, 0.7, np.where(diff < -threshold, 0.3, 0.5))
        w_cv = 1.0 - w_full
        merged = pd.DataFrame(
            w_full[:, None] * full_valid_proba.values + w_cv[:, None] * cv_valid_proba.values,
            columns=class_names,
            index=full_valid_proba.index
        )
    else:
        raise ValueError(f"Unknown rule: {rule_name}")

    pred = predict_from_proba(merged)
    score = accuracy_score(y_valid, pred)
    return score, merged

merge_rules = [
    "fixed_05_05",
    "fixed_04_06_cv",
    "dynamic_max",
    "dynamic_margin",
    "gated_max",
    "gated_margin"
]

best_rule = None
best_score = -1.0

for rule in merge_rules:
    score, _ = evaluate_merge_rule(full_valid_proba, cv_valid_proba, y_valid, rule, class_names)
    if score > best_score:
        best_score = score
        best_rule = rule

final_validation_score, _ = evaluate_merge_rule(full_valid_proba, cv_valid_proba, y_valid, best_rule, class_names)
print(f"Final Validation Performance: {final_validation_score}")

final_full_model = CatBoostClassifier(**base_model_params)
final_full_model.fit(
    X,
    y,
    cat_features=cat_cols
)

final_class_names = list(final_full_model.classes_)

full_test_proba_mean = pd.DataFrame(
    final_full_model.predict_proba(X_test),
    columns=final_class_names
)

final_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cv_test_proba_sum = pd.DataFrame(
    np.zeros((len(X_test), len(final_class_names))),
    columns=final_class_names
)

for fold, (tr_idx, va_idx) in enumerate(final_skf.split(X, y), 1):
    X_tr = X.iloc[tr_idx]
    y_tr = y.iloc[tr_idx]
    X_va = X.iloc[va_idx]
    y_va = y.iloc[va_idx]

    fold_model = CatBoostClassifier(**base_model_params)
    fold_model.fit(
        X_tr,
        y_tr,
        cat_features=cat_cols,
        eval_set=(X_va, y_va),
        use_best_model=True,
        early_stopping_rounds=50
    )

    fold_classes = list(fold_model.classes_)
    fold_test_proba = pd.DataFrame(
        fold_model.predict_proba(X_test),
        columns=fold_classes
    ).reindex(columns=final_class_names, fill_value=0.0)

    cv_test_proba_sum += fold_test_proba

cv_test_proba_mean = cv_test_proba_sum / final_skf.n_splits

def apply_merge_rule_to_test(full_test_proba_mean, cv_test_proba_mean, rule_name, final_class_names):
    if rule_name == "fixed_05_05":
        merged = 0.5 * full_test_proba_mean + 0.5 * cv_test_proba_mean
    elif rule_name == "fixed_04_06_cv":
        merged = 0.4 * full_test_proba_mean + 0.6 * cv_test_proba_mean
    elif rule_name == "dynamic_max":
        conf_full = row_confidence_max(full_test_proba_mean)
        conf_cv = row_confidence_max(cv_test_proba_mean)
        w_full = conf_full / (conf_full + conf_cv + 1e-12)
        w_full = 0.25 + 0.5 * w_full
        w_cv = 1.0 - w_full
        merged = pd.DataFrame(
            w_full[:, None] * full_test_proba_mean.values + w_cv[:, None] * cv_test_proba_mean.values,
            columns=final_class_names
        )
    elif rule_name == "dynamic_margin":
        conf_full = row_confidence_margin(full_test_proba_mean)
        conf_cv = row_confidence_margin(cv_test_proba_mean)
        w_full = conf_full / (conf_full + conf_cv + 1e-12)
        w_full = 0.25 + 0.5 * w_full
        w_cv = 1.0 - w_full
        merged = pd.DataFrame(
            w_full[:, None] * full_test_proba_mean.values + w_cv[:, None] * cv_test_proba_mean.values,
            columns=final_class_names
        )
    elif rule_name == "gated_max":
        conf_full = row_confidence_max(full_test_proba_mean)
        conf_cv = row_confidence_max(cv_test_proba_mean)
        diff = conf_full - conf_cv
        threshold = 0.05
        w_full = np.where(diff > threshold, 0.7, np.where(diff < -threshold, 0.3, 0.5))
        w_cv = 1.0 - w_full
        merged = pd.DataFrame(
            w_full[:, None] * full_test_proba_mean.values + w_cv[:, None] * cv_test_proba_mean.values,
            columns=final_class_names
        )
    elif rule_name == "gated_margin":
        conf_full = row_confidence_margin(full_test_proba_mean)
        conf_cv = row_confidence_margin(cv_test_proba_mean)
        diff = conf_full - conf_cv
        threshold = 0.03
        w_full = np.where(diff > threshold, 0.7, np.where(diff < -threshold, 0.3, 0.5))
        w_cv = 1.0 - w_full
        merged = pd.DataFrame(
            w_full[:, None] * full_test_proba_mean.values + w_cv[:, None] * cv_test_proba_mean.values,
            columns=final_class_names
        )
    else:
        raise ValueError(f"Unknown rule: {rule_name}")

    return merged

final_test_proba = apply_merge_rule_to_test(full_test_proba_mean, cv_test_proba_mean, best_rule, final_class_names)
test_pred = final_test_proba.idxmax(axis=1).values

submission = pd.DataFrame({
    ID: test[ID],
    TARGET: test_pred
})

os.makedirs("./final", exist_ok=True)
submission.to_csv("./final/submission.csv", index=False)
