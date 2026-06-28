# Tuning template (DataOps)

Minimal shape for terminal `tune_implement` scripts. **Copy your given previous structural solution verbatim**, then inject `choose_*` only on the plan's focus block.

## Rules
- Holdout only: same `train_test_split` as structural; `search.fit({"data": train_part})`; holdout eval via `search.best_learner_.predict({"data": valid_part})`. No `test_df`, no full-train refit.
- Reuse the structural solution's FE helpers, encoders, ensemble blend, and metric — change only tunable literals.
- FE via plain `def fe_func(df): ...` + `.skb.apply_func(fe_func)` — same function names as structural.
- Keep the same estimator classes as structural (e.g. if structural uses LGBM+CatBoost, do not swap to RF).
- If structural scores an ensemble, the search script must score with the **same blend** after search (tune one leg, keep others fixed).
- Reduce boosted-tree `iterations`/`n_estimators` to ~1/2 structural during search; then restore full capacity in bake.
- Print holdout score and `TUNING_BEST_PARAMS` (see below).

## Skeleton (sklearn-API estimator — inline `choose_*`)
```python
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
# model imports + metric_fn — match structural solution

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=n, random_state=m
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def fe_func(df):  # copy from structural if present
    out = df.copy()
    return out


data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(fe_func)  # omit if structural has no FE

X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(...)  # match structural encoder config
pred = X_train.skb.apply(vectorizer).skb.apply(
    YourSklearnEstimator(
        n_estimators=n_estimators,  # reduced for search budget
        learning_rate=skrub.choose_float(low_float, high_float, log=True, default=default_float, name="lr"),
        depth=skrub.choose_int(low_int, high_int, log=True, default=default_int, name="depth"),
    ),
    y=y_train,
)

search = pred.skb.make_randomized_search(n_iter=n_iter, n_jobs=1, random_state=n, fitted=True)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
score = metric_fn(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {score}")

best_params = {"lr": ...}  # map from search.best_params_ to plan param names
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
```

## Pattern: encoder focus (`TableVectorizer`)
Load `encoding_skrub.md` + Pattern 2b in `choices_hparam_pattern.md`. Copy structural FE + model; inject `choose_*` only on the vectorizer step.

```python
vectorizer = skrub.choose_from(
    {
        "default": skrub.TableVectorizer(),
        "drop_high": skrub.TableVectorizer(high_cardinality="drop"),
    },
    name="encoder_variant",
)
pred = X_train.skb.apply_func(prep).skb.apply(vectorizer).skb.apply(
    RandomForestClassifier(...), y=y_train,
)
search = pred.skb.make_randomized_search(n_iter=4, n_jobs=1, random_state=42, fitted=True)
search.fit({"data": train_part})
# ... holdout predict, TUNING_BEST_PARAMS from search.results_.iloc[0]["encoder_variant"]
```

Do **not** use `low_cardinality="one-hot"` or `"auto"` — invalid. Use `"drop"`/`"passthrough"` or transformer instances.

## Pattern: non-sklearn estimator (CatBoost, etc.)
Use a small `choose_from` variant grid — see `choices_hparam_pattern.md` Pattern 4. Do not put `choose_*` in CatBoost constructor kwargs.

## Pattern: ensemble structural
Copy both legs and the blend from structural. Inject `choose_*` only on the tunable leg; keep the other leg fixed; score with the same ensemble rule as structural:
```python
# ... same FE + split as structural ...
lgbm_pred = X_train.skb.apply(vec_lgbm).skb.apply(lgbm_with_choose, y=y_train)
cat_pred = X_train.skb.apply(vec_cat).skb.apply(cat_fixed, y=y_train)

search = lgbm_pred.skb.make_randomized_search(n_iter=n_iter, n_jobs=1, random_state=n, fitted=True)
search.fit({"data": train_part})

valid_lgbm = search.best_learner_.predict({"data": valid_part})
cat_learner = cat_pred.skb.make_learner(fitted=True)
valid_cat = cat_learner.predict({"data": valid_part})
valid_pred = blend(valid_lgbm, valid_cat)  # same rule as structural
print(f"Final Validation Performance: {metric_fn(...)}")
```

## Required output
```
Final Validation Performance: <value>
TUNING_BEST_PARAMS: {"param_name": ...}
```

Load `references/choices_hparam_pattern.md` for search execution; load `references/encoding_skrub.md` when tuning encoders.
