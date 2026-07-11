
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from catboost import CatBoostClassifier
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "catboost is not installed. Please install it before running this script: pip install catboost"
    )

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
    "MTRANS",
]

X = train.drop(columns=[ID, TARGET])
y = train[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

def prepare_features(X_tr, X_va, X_te, use_cat_features):
    if use_cat_features:
        return X_tr.copy(), X_va.copy(), X_te.copy(), cat_cols

    X_tr_enc = pd.get_dummies(X_tr, columns=cat_cols, drop_first=False)
    X_va_enc = pd.get_dummies(X_va, columns=cat_cols, drop_first=False)
    X_te_enc = pd.get_dummies(X_te, columns=cat_cols, drop_first=False)

    all_cols = X_tr_enc.columns.union(X_va_enc.columns).union(X_te_enc.columns)
    X_tr_enc = X_tr_enc.reindex(columns=all_cols, fill_value=0)
    X_va_enc = X_va_enc.reindex(columns=all_cols, fill_value=0)
    X_te_enc = X_te_enc.reindex(columns=all_cols, fill_value=0)

    return X_tr_enc, X_va_enc, X_te_enc, None

def build_model(depth):
    return CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="Accuracy",
        iterations=700,
        learning_rate=0.05,
        depth=depth,
        random_seed=42,
        verbose=0,
        allow_writing_files=False,
    )

def run_experiment(name, use_cat_features=True, use_early_stopping=True, depth=6):
    X_tr_use, X_va_use, _, cat_features_for_fit = prepare_features(
        X_train, X_valid, test.drop(columns=[ID]), use_cat_features
    )

    model = build_model(depth)

    fit_params = {}
    if use_cat_features:
        fit_params["cat_features"] = cat_features_for_fit
    if use_early_stopping:
        fit_params["eval_set"] = (X_va_use, y_valid)
        fit_params["use_best_model"] = True
        fit_params["early_stopping_rounds"] = 100

    model.fit(X_tr_use, y_train, **fit_params)
    pred = model.predict(X_va_use)
    pred = pd.Series(pred.reshape(-1)).astype(str).values
    score = accuracy_score(y_valid, pred)
    print(f"{name}: validation_accuracy={score:.6f}")
    return score

results = {}

results["baseline"] = run_experiment(
    name="baseline",
    use_cat_features=True,
    use_early_stopping=True,
    depth=6,
)

results["no_cat_features"] = run_experiment(
    name="ablation_no_cat_features",
    use_cat_features=False,
    use_early_stopping=True,
    depth=6,
)

results["no_early_stopping"] = run_experiment(
    name="ablation_no_early_stopping",
    use_cat_features=True,
    use_early_stopping=False,
    depth=6,
)

results["shallower_depth"] = run_experiment(
    name="ablation_shallower_depth_4",
    use_cat_features=True,
    use_early_stopping=True,
    depth=4,
)

baseline_score = results["baseline"]
ablations_only = {k: v for k, v in results.items() if k != "baseline"}
drops = {k: baseline_score - v for k, v in ablations_only.items()}

print("\nPerformance change vs baseline:")
for name, drop in drops.items():
    print(f"{name}: accuracy_drop={drop:.6f}")

most_important_part = max(drops, key=drops.get)
print(
    f"\nBiggest contributor to performance: {most_important_part} "
    f"(largest accuracy drop = {drops[most_important_part]:.6f})"
)

best_config_name = max(results, key=results.get)
if best_config_name == "baseline":
    best_use_cat_features = True
    best_use_early_stopping = True
    best_depth = 6
elif best_config_name == "no_cat_features":
    best_use_cat_features = False
    best_use_early_stopping = True
    best_depth = 6
elif best_config_name == "no_early_stopping":
    best_use_cat_features = True
    best_use_early_stopping = False
    best_depth = 6
else:
    best_use_cat_features = True
    best_use_early_stopping = True
    best_depth = 4

final_validation_score = results[best_config_name]
print(f"Final Validation Performance: {final_validation_score}")

X_full = train.drop(columns=[ID, TARGET])
y_full = train[TARGET]
X_test = test.drop(columns=[ID])

if best_use_early_stopping:
    X_tr_final, X_va_final, y_tr_final, y_va_final = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=42,
        stratify=y_full,
    )
    X_tr_final_use, X_va_final_use, X_test_final_use, cat_features_final = prepare_features(
        X_tr_final, X_va_final, X_test, best_use_cat_features
    )
    model = build_model(best_depth)
    final_fit_params = {}
    if best_use_cat_features:
        final_fit_params["cat_features"] = cat_features_final
    final_fit_params["eval_set"] = (X_va_final_use, y_va_final)
    final_fit_params["use_best_model"] = True
    final_fit_params["early_stopping_rounds"] = 100
    model.fit(X_tr_final_use, y_tr_final, **final_fit_params)
    test_pred = model.predict(X_test_final_use)
else:
    X_full_use, _, X_test_use, cat_features_for_final = prepare_features(
        X_full, X_full.iloc[:1].copy(), X_test, best_use_cat_features
    )
    model = build_model(best_depth)
    final_fit_params = {}
    if best_use_cat_features:
        final_fit_params["cat_features"] = cat_features_for_final
    model.fit(X_full_use, y_full, **final_fit_params)
    test_pred = model.predict(X_test_use)

test_pred = pd.Series(test_pred.reshape(-1)).astype(str).values

submission = pd.DataFrame({
    ID: test[ID],
    TARGET: test_pred,
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
