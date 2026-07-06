# Tuning template (DataOps)

Minimal shape for terminal `tune_implement` scripts. **Copy your given previous structural solution verbatim**, then inject `choose_*` only on the plan's focus block.

## Rules
- Holdout only: same `train_test_split` as structural; `search.fit({"data": train_part})`; holdout eval via `search.best_learner_.predict({"data": valid_part})`. No `test_df`, no full-train refit.
- Reuse the structural solution's FE helpers, encoders, ensemble blend, and metric — change only tunable literals.
- FE via plain `def fe_func(df): ...` + `.skb.apply_func(fe_func)` — same function names as structural.
- Keep the same estimator classes as structural (e.g. if structural uses LGBM+CatBoost, do not swap to RF).
- If structural scores an ensemble, the search script must score with the **same blend** after search (tune one leg, keep others fixed).
- Reduce boosted-tree `iterations`/`n_estimators` to ~1/2 structural during search; then restore full capacity in bake.
- Search parallelism: search `n_jobs` parallelizes trials; tree boosters thread inside each fit — default search `n_jobs=1` for LightGBM/CatBoost/XGBoost or estimators with `n_jobs`≠1 (predictable runtime). Search `n_jobs=2` is OK with estimator `n_jobs=1` or very cheap trials. Single-threaded sklearn: `n_jobs=2` (up to `4` if trials are very fast). Never search `n_jobs=-1`.
- Print holdout score and `TUNING_BEST_PARAMS` (see below).

## Standard Pattern Skeleton (if sklearn-API estimator — use inline `choose_*`)
```python
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
# model imports + metric_fn — match structural solution

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
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
    YourEstimator(
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

# see TUNING_BEST_PARAMS mapping section below
best_params = {...}
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
```

## Pattern: encoder focus (`TableVectorizer`)
Load `encoding_skrub.md` + Pattern 2b in `choices_hparam_pattern.md`. Copy structural FE + model; inject `choose_*` only on the vectorizer step.

```python
vectorizer = skrub.choose_from(
    {
        "default": skrub.TableVectorizer(),
        "drop_high": skrub.TableVectorizer(high_cardinality="drop"),
        "pass_low": skrub.TableVectorizer(high_cardinality="passthrough"),
    },
    name="encoder_variant",
)
pred = X_train.skb.apply_func(prep).skb.apply(vectorizer).skb.apply(
    YourModel(...), y=y_train,
)
search = pred.skb.make_randomized_search(n_iter=n_iter, n_jobs=n_jobs, random_state=random_state, fitted=True)
search.fit({"data": train_part})
# ... holdout predict, TUNING_BEST_PARAMS from search.results_.iloc[0]["encoder_variant"]
```

Do **not** use `low_cardinality="one-hot"` or `"auto"` — invalid. Use `"drop"`/`"passthrough"` or you can also use `choose_*` to search over transformer instances.

## Pattern 2: non-sklearn estimator (CatBoost, etc.)
Inline `choose_*` in constructor kwargs does **not** resolve — use a small `choose_from` variant grid (whole pre-built estimator per key):
```python
cat_variants = {
    "d6_lr0.03": dict(depth=depth_1, learning_rate=lr_1, iterations=iterations, verbose=-1),
    "d8_lr0.03": dict(depth=depth_2, learning_rate=lr_1, iterations=iterations, verbose=-1),
    "d8_lr0.05": dict(depth=depth_2, learning_rate=lr_2, iterations=iterations, verbose=-1),
}
cat_model = skrub.choose_from(
    {k: CatBoostClassifier(**p, random_seed=random_state) for k, p in cat_variants.items()},
    name="cat_variant",
)
pred = X_train.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
search = pred.skb.make_randomized_search(n_iter=n_iter, n_jobs=n_jobs, random_state=random_state, fitted=True)
search.fit({"data": train_part})
chosen = search.results_.iloc[0]["cat_variant"]
best_params = dict(cat_variants[chosen])
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
```
LightGBM/XGBoost are sklearn-API — use Pattern 1 inline `choose_*` instead. Details: `choices_hparam_pattern.md` Pattern 4.

## Pattern 3: ensemble structural
Requires Pattern A (multi-leg). See `ensemble_dataops_patterns.md`. Copy legs and blend from structural; inject `choose_*` only on the tunable leg; keep others fixed:
```python
# ... same FE + split as structural ...
lgbm_pred = X_train.skb.apply(vec_lgbm).skb.apply(lgbm_with_choose, y=y_train)
cat_pred = X_train.skb.apply(vec_cat).skb.apply(cat_fixed, y=y_train)

search = lgbm_pred.skb.make_randomized_search(n_iter=n_iter, n_jobs=n_jobs, random_state=n, fitted=True)
search.fit({"data": train_part})

valid_lgbm = search.best_learner_.predict({"data": valid_part})
cat_learner = cat_pred.skb.make_learner(fitted=True)
valid_cat = cat_learner.predict({"data": valid_part})
valid_pred = blend(valid_lgbm, valid_cat)  # same rule as structural
print(f"Final Validation Performance: {metric_fn(...)}")
```

## TUNING_BEST_PARAMS mapping
- **Never** map by `data_op__0`, `data_op__1`, ... — index order is unstable and causes swaps.
- **Inline `choose_*`:** for each plan `tunable_params[]` entry, pick the value from `search.best_params_.values()` whose numeric range matches the spec (`choose_int` / `choose_float`); fall back to spec `default`. Key output by plan `name`.
- **`choose_from` variant grid:** `chosen = search.results_.iloc[0]["<choice_name>"]`; `best_params = dict(variants[chosen])`.
- Sanity-check before print: ints in range, floats in `(0, 1]` for learning rates, no swapped param types.

```python
# Map search values → plan param names (never by data_op__ index)
tune_plan = {"tunable_params": [{"name": "lr", "kind": "choose_float", "low": 0.01, "high": 0.04, "default": 0.02}, ...]}
raw_values = list(search.best_params_.values())
best_params = {}
remaining = list(raw_values)
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
```

## Required output
```
Final Validation Performance: <value>
TUNING_BEST_PARAMS: {"param_name": ...}
```

## When to load other references
- Load `choices_hparam_pattern.md` for search patterns and tune rules.
- Load `encoding_skrub.md` when tuning encoders.
- Load `ensemble_dataops_patterns.md` when the structural solution is multi-leg.
- Load `dataops_api_quickmap.md` for holdout bind and learner contracts.
