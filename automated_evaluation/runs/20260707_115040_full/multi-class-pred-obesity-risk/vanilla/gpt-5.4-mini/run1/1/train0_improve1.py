

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

target = "NObeyesdad"
y = train[target]
X = train.drop(columns=[target])

# Keep identifier separate from features
test_id = test["id"].copy()

# Basic preprocessing for target encoding
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols and c != "id"]

# Use only feature columns
X_feat = X.drop(columns=["id"], errors="ignore").copy()
test_feat = test.drop(columns=["id"], errors="ignore").copy()

# Label encoding for target
classes = np.sort(y.unique())
class_to_int = {c: i for i, c in enumerate(classes)}
y_int = y.map(class_to_int).values
n_classes = len(classes)

# Leakage-safe CV target encoding with simple smoothing
n_splits = 5
alpha = 20.0
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

global_means = np.bincount(y_int, minlength=n_classes) / len(y_int)

X_te = pd.DataFrame(index=X_feat.index)
test_te = pd.DataFrame(index=test_feat.index)

for col in cat_cols:
    oof_col = np.zeros((len(X_feat), n_classes), dtype=np.float32)
    test_col_accum = np.zeros((len(test_feat), n_classes), dtype=np.float32)

    for tr_idx, val_idx in skf.split(X_feat, y_int):
        X_tr_fold = X_feat.iloc[tr_idx]
        y_tr_fold = y_int[tr_idx]

        stats = pd.DataFrame({"cat": X_tr_fold[col].astype(str), "y": y_tr_fold})
        count_table = stats.groupby("cat")["y"].agg(["count"])
        class_tables = {
            k: stats.assign(cls=(stats["y"] == k).astype(np.float32)).groupby("cat")["cls"].mean()
            for k in range(n_classes)
        }

        tr_counts = count_table["count"]
        for k in range(n_classes):
            means = class_tables[k]
            enc_map = ((means * tr_counts.reindex(means.index).values) + alpha * global_means[k]) / (
                tr_counts.reindex(means.index).values + alpha
            )
            val_cat = X_feat.iloc[val_idx][col].astype(str)
            oof_col[val_idx, k] = val_cat.map(enc_map).fillna(global_means[k]).values.astype(np.float32)

    # Full-fit encoder for test-time transform
    stats_full = pd.DataFrame({"cat": X_feat[col].astype(str), "y": y_int})
    count_full = stats_full.groupby("cat")["y"].agg(["count"])
    tr_counts_full = count_full["count"]

    for k in range(n_classes):
        means_full = stats_full.assign(cls=(stats_full["y"] == k).astype(np.float32)).groupby("cat")["cls"].mean()
        enc_map_full = ((means_full * tr_counts_full.reindex(means_full.index).values) + alpha * global_means[k]) / (
            tr_counts_full.reindex(means_full.index).values + alpha
        )
        X_te[col + f"_te_{k}"] = pd.Series(oof_col[:, k], index=X_feat.index)
        test_te[col + f"_te_{k}"] = test_feat[col].astype(str).map(enc_map_full).fillna(global_means[k]).values.astype(np.float32)

# Combine numeric features + target encodings
X_num = pd.concat([X_feat[num_cols].reset_index(drop=True), X_te.reset_index(drop=True)], axis=1)
test_num = pd.concat([test_feat[num_cols].reset_index(drop=True), test_te.reset_index(drop=True)], axis=1)

# Handle missing values
X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(-999)
test_num = test_num.replace([np.inf, -np.inf], np.nan).fillna(-999)

# Holdout split for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_num, y_int, test_size=0.2, random_state=42, stratify=y_int
)

seeds = [42, 52, 62]
test_preds = []
val_preds = []

for seed in seeds:
    model = CatBoostClassifier(
        loss_function="MultiClass",
        iterations=4000,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=8.0,
        random_seed=seed,
        eval_metric="Accuracy",
        od_type="Iter",
        od_wait=150,
        verbose=200
    )

    model.fit(
        X_tr,
        y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    val_pred = model.predict(X_val).astype(int).ravel()
    val_preds.append(val_pred)

    test_proba = model.predict_proba(test_num)
    test_preds.append(test_proba)

# Majority vote on validation ensemble for a quick score
val_preds = np.stack(val_preds, axis=1)
ensemble_val_pred = np.apply_along_axis(
    lambda row: np.bincount(row, minlength=n_classes).argmax(),
    axis=1,
    arr=val_preds
)
final_validation_score = accuracy_score(y_val, ensemble_val_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Average probabilities for test-time prediction, then take argmax
avg_test_proba = np.mean(np.stack(test_preds, axis=0), axis=0)
test_pred_int = np.argmax(avg_test_proba, axis=1)
test_pred = pd.Series(test_pred_int).map({i: c for i, c in enumerate(classes)}).values

submission = pd.DataFrame({
    "id": test_id,
    "NObeyesdad": test_pred
})
submission.to_csv("submission.csv", index=False)
