import pandas as pd
import numpy as np
import skrub
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

TRAIN_PATH = "./input/train.csv"
TARGET_COL = "median_house_value"

train_df = pd.read_csv(TRAIN_PATH)

if len(train_df) > 20000:
    train_df = train_df.sample(n=20000, random_state=42).reset_index(drop=True)

rng = np.random.RandomState(42)
idx = np.arange(len(train_df))
rng.shuffle(idx)
split = int(0.8 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_split = train_df.iloc[tr_idx].reset_index(drop=True)
valid_split = train_df.iloc[va_idx].reset_index(drop=True)


def fit_predict_rmse(train_part, valid_part, use_lgbm=True, use_hgb=True, use_ensemble=True):
    preds = []
    names = []

    if use_lgbm:
        data = skrub.var("data", train_part)
        X = data.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
        y = data[TARGET_COL].skb.mark_as_y()

        vectorizer = skrub.TableVectorizer()
        model = LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.03,
            num_leaves=64,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        graph = X.skb.apply(vectorizer).skb.apply(model, y=y)
        learner = graph.skb.make_learner(fitted=True)
        learner.fit({"data": train_part})
        preds.append(learner.predict({"data": valid_part}))
        names.append("LGBM")

    if use_hgb:
        data2 = skrub.var("data", train_part)
        X2 = data2.drop(columns=[TARGET_COL], errors="ignore").skb.mark_as_X()
        y2 = data2[TARGET_COL].skb.mark_as_y()

        vectorizer2 = skrub.TableVectorizer()
        model2 = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=6,
            max_iter=300,
            random_state=42,
        )
        graph2 = X2.skb.apply(vectorizer2).skb.apply(model2, y=y2)
        learner2 = graph2.skb.make_learner(fitted=True)
        learner2.fit({"data": train_part})
        preds.append(learner2.predict({"data": valid_part}))
        names.append("HGB")

    if use_ensemble and len(preds) == 2:
        valid_pred = 0.5 * preds[0] + 0.5 * preds[1]
        variant_name = "Ensemble(LGBM+HGB)"
    elif len(preds) == 1:
        valid_pred = preds[0]
        variant_name = names[0]
    else:
        raise ValueError("Invalid ablation configuration.")

    rmse = mean_squared_error(valid_part[TARGET_COL], valid_pred) ** 0.5
    return variant_name, rmse


results = {}

variant_name, rmse = fit_predict_rmse(train_split, valid_split, use_lgbm=True, use_hgb=True, use_ensemble=True)
results["full_ensemble"] = rmse
print(f"Ablation[full_ensemble] RMSE: {rmse}")

variant_name, rmse = fit_predict_rmse(train_split, valid_split, use_lgbm=True, use_hgb=False, use_ensemble=False)
results["lgbm_only"] = rmse
print(f"Ablation[lgbm_only] RMSE: {rmse}")

variant_name, rmse = fit_predict_rmse(train_split, valid_split, use_lgbm=False, use_hgb=True, use_ensemble=False)
results["hgb_only"] = rmse
print(f"Ablation[hgb_only] RMSE: {rmse}")

best_variant = min(results, key=results.get)
best_score = results[best_variant]
worst_variant = max(results, key=results.get)
worst_score = results[worst_variant]

print(f"Final Validation Performance: {results['full_ensemble']}")
print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")
print(f"Worst ablation variant: {worst_variant} | RMSE: {worst_score}")

if best_variant == "full_ensemble":
    print("Most contributing part: keeping both models and ensembling them.")
elif best_variant == "lgbm_only":
    print("Most contributing part: the LightGBM branch contributes the most.")
elif best_variant == "hgb_only":
    print("Most contributing part: the HistGradientBoosting branch contributes the most.")