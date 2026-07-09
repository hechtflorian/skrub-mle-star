# Choice-Based Tuning Pattern (DataOps)

Use this reference when adding tunable parameters or model/encoder alternatives inside a DataOps graph.

## APIs quickmap
- `skrub.choose_int(low, high, n_steps=..., log=..., default=..., name=...)`
- `skrub.choose_float(low, high, log=True, default=..., name=...)`
- `skrub.choose_from(outcomes: list or dict, name=...)` — list or dict; dict keys must be strings
- `.skb.describe_param_grid()`
- `.skb.make_randomized_search(...)`
- `.skb.make_grid_search(...)`
- `.skb.make_learner(...)`

## Critical Rules (must follow)
### Critical rule 1: `choose_*` defaults are not tuning
- `choose_*` nodes define a search space, but no search happens unless you run `.skb.make_randomized_search(...)`.
- Calling `.skb.make_learner(...)` on a graph that contains `choose_*` uses default choice values only.
- This default behavior is valid for a quick baseline, but it must not be presented as tuned.
- If you are not running search, replace `choose_*` with explicit fixed constants in final training code.

### Critical rule 2: `choose_from` dict keys must be strings
- Valid:
```python
max_depth = skrub.choose_from({"6": 6, "8": 8, "10": 10}, name="max_depth")
```
- Invalid (will raise type error):
```python
max_depth = skrub.choose_from({6: 6, 8: 8, 10: 10}, name="max_depth")
```
- If you only need an integer or float range, prefer the following:
```python
max_depth = skrub.choose_int(6, 10, name="max_depth")
max_depth = skrub.choose_float(0.5, 1.0, name="lr")
```
- List form is also valid: `skrub.choose_from([0.1, 1.0, 10.0], name="alpha")`

### Critical rule 3: do not mix train / val / full data
- `skrub.var("data", train_df)` binds the pipeline to **full** training data.
- `pred.skb.make_learner(fitted=True)` fits on that bound data (full `train_df`), even if you later `predict({"data": valid_part})` for scoring — **this leaks validation rows into training**.
- For the `Final Validation Performance` line in fixed-parameter scripts, bind `train_part` only (see `dataops_api_quickmap.md` holdout section).
- `search.fit({"data": train_part})` fits search on **only** `train_part`; holdout eval must use `search.best_learner_.predict({"data": valid_part})`.
- These paths produce **different scores**. For MLE-STAR tune stages, copy the structural solution's exact split, then use the **same fit + eval path** in `tune_implement` and `tune_bake`.

### Critical rule 4: inline `choose_*` in constructor kwargs works only for sklearn-API estimators
- skrub substitutes `choose_*` placed inside an estimator's constructor kwargs **only when the estimator subclasses `sklearn.base.BaseEstimator`** (all sklearn models, LightGBM `LGBM*`, XGBoost `XGB*`).
- For estimators that are **not** `BaseEstimator` subclasses (e.g. `catboost.CatBoostRegressor`), the choice object is never resolved, reaches `fit` raw, and crashes (CatBoost: `TypeError: Object of type NumericChoice is not JSON serializable`). Patterns 1/3 inline kwargs cannot work there — use Pattern 4.
- Quick check when unsure: `from sklearn.base import BaseEstimator; isinstance(est, BaseEstimator)`.

### Critical rule 5: search must fit the execution time budget
- Total search runtime ≈ (n_iter + 1) × single-fit time; the whole script must finish **well under the execution timeout (default 600s)**. If one structural fit takes minutes, a full-capacity search will time out and the tuning stage fails.
- For boosted trees, **reduce capacity during search**: cut `iterations`/`n_estimators` to roughly 1/4 of the structural value or use early stopping. Relative ranking of nearby configs is preserved; the winner is baked at structural capacity afterwards.
- Search `n_jobs` runs trials in parallel; multithreaded estimators (e.g. CatBoost, LightGBM, XGBoost) or sklearn with `n_jobs`≠1 also parallelize each fit — default search `n_jobs=1` to avoid CPU oversubscription and stay under the exec timeout (not because higher values fail). Search `n_jobs=2` is OK if the estimator uses `n_jobs=1` or trials are very cheap. For single-threaded sklearn, `n_jobs=2` is reasonable; up to `4` only for very fast fits. Never search `n_jobs=-1`.
- Keep the search space small and focused: few params, tight ranges, low `n_iter`. One cheap completed search beats an ambitious one that times out.

## Pattern 1: tune scalar hyperparameters in place
```python
import pandas as pd
import skrub

data = pd.read_csv(example_csv_path).sample(frac=1.0, random_state=random_state)
X = skrub.X(data[["text"]])
y = skrub.y(data["is_toxic"])

encoder = skrub.MinHashEncoder(
    n_components=skrub.choose_int(5, 15, n_steps=5, name="N components")
)
classifier = YourClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)

pred = X.skb.apply(encoder).skb.apply(classifier, y=y)
#print(pred.skb.describe_param_grid())
```

## Pattern 2: choose between encoder families
```python
n_components = skrub.choose_int(5, 15, name="N components")
encoder = skrub.choose_from(
    {
        "minhash": skrub.MinHashEncoder(n_components=n_components), # your_encoder_1
        "lse": skrub.StringEncoder(n_components=n_components),  # your_encoder_2
    },
    name="encoder",
)
pred = X.skb.apply(encoder).skb.apply(classifier, y=y)
```

## Pattern 2b: tune `TableVectorizer` (encoder focus block)
`TableVectorizer` `low_cardinality` / `high_cardinality` accept **`"passthrough"`**, **`"drop"`**, or a **transformer instance** — not `"one-hot"`, `"auto"`, etc.

**Variant grid (whole vectorizer):**
```python
vectorizer = skrub.choose_from(
    {
        "default": skrub.TableVectorizer(),
        "drop_high": skrub.TableVectorizer(high_cardinality="drop"),
    },
    name="encoder_variant",
)
pred = X.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(model, y=y)
```

**Inline encoder choice on `high_cardinality=`**:
```python
n = skrub.choose_int(5, 15, name="n_components")
encoder = skrub.choose_from(
    {"minhash": skrub.MinHashEncoder(n_components=n),   # your_encoder_1
     "string": skrub.StringEncoder(n_components=n)},    # your_encoder_2
    name="encoder",
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder)
pred = X.skb.apply(vectorizer).skb.apply(model, y=y)
```

For holdout search + bake handoff on Pattern 2b, load `encoding_skrub.md` for `TUNING_BEST_PARAMS` mapping via `search.results_.iloc[0]["encoder_variant"]`.

## Pattern 3: choose between model families
```python
from sklearn.some_model import YourModel 

model1 = YourModel1(learning_rate=skrub.choose_float(low_float, high_float, log=True, name="your_model_param"))
model2 = YourModel2(alpha=skrub.choose_float(low_float, high_float, log=True, name="your_model_param"))
classifier = skrub.choose_from({"model1": model1, "model2": model2}, name="classifier")
pred = X.skb.apply(encoder).skb.apply(classifier, y=y)
```

## Pattern 4: tune non-sklearn estimators via a `choose_from` variant grid
Select the **whole pre-configured estimator** instead of per-param choices. Keep the grid small (~2x `n_iter` combos), string keys.
```python
variants = {
    "d7_lr0.03": dict(depth=depth1, learning_rate=lr1),
    "d8_lr0.03": dict(depth=depth2, learning_rate=lr1),
    "d8_lr0.05": dict(depth=depth2, learning_rate=lr2),
    "d9_lr0.05": dict(depth=depth3, learning_rate=lr2),
}
model = skrub.choose_from(
    {k: YourNonSklearnModel(**p, random_seed=random_state, verbose=0) for k, p in variants.items()},
    name="model_variant",
)
pred = X.skb.apply(vectorizer).skb.apply(model, y=y)
search = pred.skb.make_randomized_search(n_iter=n_iter, random_state=random_state, fitted=True)
search.fit({"data": train_part})

chosen = search.results_.iloc[0]["model_variant"]  # results_ row 0 = best; column name = choice name
best_params = dict(variants[chosen])               # literal params -> clean bake handoff
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
```

## Search execution pattern
Holdout search (`tune_implement` only — holdout metric + `TUNING_BEST_PARAMS`):
```python
import json

search = pred.skb.make_randomized_search(
    n_iter=n_iter, n_jobs=n_jobs, random_state=n, fitted=True
)  # single-threaded sklearn; for LGBM/CatBoost/XGBoost default search n_jobs=1 (or estimator n_jobs=1 + search n_jobs=2)
search.fit({"data": train_part})  # train fold only — not full train_df, not valid_part
best_learner = search.best_learner_

valid_pred = best_learner.predict({"data": valid_part})
holdout_score = your_metric_fn(valid_part[target_col].values, valid_pred)
print(f"Final Validation Performance: {holdout_score}")

#print(pred.skb.describe_param_grid())  # inspect param names before bake, returns string
best_params = {}
for spec in tune_plan["tunable_params"]:  # use plan param names, not data_op__ keys
    name = spec["name"]
    # map each search.best_params_ value to the matching spec by kind/range/outcomes
    ...
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
```
Tune agents should prefer explicit holdout `search.fit` above when structural code splits train/val.

## Anti-pattern vs correct pattern
- Anti-pattern (holdout leakage): `skrub.var("data", train_df)` + split + `make_learner(fitted=True)` + `predict({"data": valid_part})` for the metric line.
- Anti-pattern (fake tuning): define `choose_*` and then call only `pred.skb.make_learner(fitted=True)`.
- Anti-pattern (unresolved choose in estimator): `lr = skrub.choose_float(...)` then `Estimator(learning_rate=lr)` — many libraries copy/serialize kwargs at init and fail on skrub choice objects; keep `choose_*` on the DataOps apply path and resolve via search.
- Correct (inline on apply, sklearn-API estimators only): `pred = X.skb.apply(Estimator(lr=skrub.choose_float(0.01, 0.1, name="lr")), y=y)` then `search = pred.skb.make_randomized_search(...)` — no intermediate variable holding a choice object. For non-sklearn estimators (e.g. CatBoost) this also fails — use Pattern 4.
- Anti-pattern (terminal tune crash): call `json.dumps(best_params)` on skrub/search params without `default=str` or numpy-to-Python conversion.
- Anti-pattern (incomparable scores): `search.fit({"data": train_part})` in tune_implement, then `make_learner(fitted=True)` on full `train_df` in tune_bake for the metric line.
- Anti-pattern (bake mapping): assume `search.best_params_` keys match `name=` strings — keys are often `data_op__0`, `data_op__1`, … and index order may not match plan order. Map **values** to plan `tunable_params` by kind/range/outcomes (or build human-named params in `tune_implement`), not by key name or positional index alone.
- Anti-pattern (encoder tune): `TableVectorizer(low_cardinality="one-hot")` or `"auto"` — use `"drop"`/`"passthrough"`, transformer instances, or a `choose_from` grid of whole vectorizers (Pattern 2b; `encoding_skrub.md`).
- Correct holdout (early stages): bind `train_part`, print metric — no test/full-train block until submission.
- Correct tuning: define `choose_*`, run search (`make_randomized_search` / `make_grid_search`), then train/predict with best search result.
- Correct fixed-parameter run: no `choose_*`; use concrete parameter values directly.

## Refinement terminal tune (search → bake handoff)
- Runs once in the dedicated `tuning` pipeline stage after refinement completes.
- Load `references/tuning_dataops_template.md` for `tune_implement` script skeleton example.
- `tune_implement` script: keep structural DataOps graph; add in-graph `choose_*` only on one focus block; run `make_randomized_search`, **`search.fit({"data": train_part})`**, holdout eval with `search.best_learner_.predict({"data": valid_part})`.
- Print best params on one line with JSON-safe serialization:
  `print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))`
- Do **not** use bare `json.dumps(best_params)` on skrub/search output — numpy scalars will crash the script.
- `tune_bake` script: replace each `choose_*` with literals from best params; **no** `choose_*` or search calls; score with Block 1 (`skrub.var("data", train_part)` + holdout predict) — same protocol as structural code; **no** `test_df` or full-train refit (will already happen at later submission stage)
- One focus block per search; keep `n_iter` low (≤4 by default). Set `verbose=-1` on LightGBM/CatBoost during search to limit stdout noise.

## Checklist
- Tunables are in-graph, not external ad-hoc parameter dicts.
- Main path remains DataOps (`.skb.apply(...)` chain).
- Search object comes from the final prediction DataOp.
- For `choose_from({...})`, dictionary keys are readable outcome names and must be strings.
- If `choose_*` appears in final code, search execution is present and best search output is used.
- `train_part`, `valid_part`, and full `train_df` are used consistently across search, bake, and structural scoring.
- `describe_param_grid()` checked; bake literals mapped from `best_params_` **values** to plan param names by kind/range, not assumed key names or creation order.

## When to load other references
- Load `dataops_api_quickmap.md` for holdout pipeline shape and fit/predict contracts.
- Load `tuning_dataops_template.md` for tune_implement script skeleton.
- Load `encoding_skrub.md` when tuning encoders or `TableVectorizer`.
- Load `skrub_subsampling.md` when search is too slow.