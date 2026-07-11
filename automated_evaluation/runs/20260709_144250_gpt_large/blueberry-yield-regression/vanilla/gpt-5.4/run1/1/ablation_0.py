
import os
import copy
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train_path = os.path.join(".", "input", "train.csv")
train = pd.read_csv(train_path)

X = train.drop(columns=["id", "yield"])
y = train["yield"]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

def evaluate_config(name, use_cat=True, use_lgb=True, use_ensemble=True, use_early_stopping=True):
    cat_pred = None
    lgb_pred = None

    if use_cat:
        cat_model = CatBoostRegressor(
            iterations=2000,
            learning_rate=0.03,
            depth=6,
            loss_function="MAE",
            eval_metric="MAE",
            random_seed=42,
            verbose=False
        )
        if use_early_stopping:
            cat_model.fit(
                X_train,
                y_train,
                eval_set=(X_valid, y_valid),
                use_best_model=True,
                verbose=False
            )
        else:
            cat_model.fit(
                X_train,
                y_train,
                verbose=False
            )
        cat_pred = cat_model.predict(X_valid)

    if use_lgb:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression_l1",
            random_state=42
        )
        if use_early_stopping:
            lgb_model.fit(
                X_train,
                y_train,
                eval_set=[(X_valid, y_valid)],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
            )
        else:
            lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_valid)

    if use_ensemble:
        preds = []
        if cat_pred is not None:
            preds.append(cat_pred)
        if lgb_pred is not None:
            preds.append(lgb_pred)
        valid_pred = np.mean(preds, axis=0)
    else:
        valid_pred = cat_pred if cat_pred is not None else lgb_pred

    mae = mean_absolute_error(y_valid, valid_pred)
    print(f"{name}: MAE={mae:.6f}")
    return mae

results = {}

results["baseline_full_ensemble"] = evaluate_config(
    name="baseline_full_ensemble",
    use_cat=True,
    use_lgb=True,
    use_ensemble=True,
    use_early_stopping=True
)

results["ablation_no_ensemble_cat_only"] = evaluate_config(
    name="ablation_no_ensemble_cat_only",
    use_cat=True,
    use_lgb=False,
    use_ensemble=False,
    use_early_stopping=True
)

results["ablation_no_ensemble_lgb_only"] = evaluate_config(
    name="ablation_no_ensemble_lgb_only",
    use_cat=False,
    use_lgb=True,
    use_ensemble=False,
    use_early_stopping=True
)

results["ablation_no_early_stopping"] = evaluate_config(
    name="ablation_no_early_stopping",
    use_cat=True,
    use_lgb=True,
    use_ensemble=True,
    use_early_stopping=False
)

baseline = results["baseline_full_ensemble"]
drops = {}

for name, score in results.items():
    if name != "baseline_full_ensemble":
        drops[name] = score - baseline

print("\nPerformance impact relative to baseline:")
for name, delta in sorted(drops.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: MAE change = {delta:+.6f}")

worst_ablation = max(drops, key=drops.get)
if "cat_only" in worst_ablation or "lgb_only" in worst_ablation:
    most_important_part = "the CatBoost + LightGBM ensemble"
elif "early_stopping" in worst_ablation:
    most_important_part = "early stopping / best-iteration selection"
else:
    most_important_part = worst_ablation

print(f"\nMost important contributing part: {most_important_part}")
