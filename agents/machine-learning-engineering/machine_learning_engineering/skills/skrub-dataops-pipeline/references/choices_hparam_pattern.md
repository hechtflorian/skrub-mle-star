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

## Critical rule: `choose_*` defaults are not tuning
- `choose_*` nodes define a search space, but no search happens unless you run `.skb.make_randomized_search(...)` or `.skb.make_grid_search(...)` (or Optuna trial flow).
- Calling `.skb.make_learner(...)` on a graph that contains `choose_*` uses default choice values only.
- This default behavior is valid for a quick baseline, but it must not be presented as tuned.
- If you are not running search, replace `choose_*` with explicit fixed constants in final training code.

## Critical rule: `choose_from` dict keys must be strings
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
max_depth = skrub.choose_float(6.0, 10.0, name="max_depth")
```
- List form is valid: `skrub.choose_from([0.1, 1.0, 10.0], name="alpha")`

## Critical rule: do not mix train / val / full data
- `skrub.var("data", train_df)` binds the pipeline to **full** training data.
- `pred.skb.make_learner(fitted=True)` fits on that bound data (full `train_df`), even if you later `predict({"data": valid_part})` for scoring — **this leaks validation rows into training**.
- For the `Final Validation Performance` line in fixed-parameter scripts, use the **two-block pattern** in `dataops_api_quickmap.md`: Block 1 binds `train_part`, Block 2 binds `train_df` for test/submission.
- `search.fit({"data": train_part})` fits search on **only** `train_part`; holdout eval must use `search.best_learner_.predict({"data": valid_part})`.
- These paths produce **different scores**. For MLE-STAR tune stages, copy the structural solution's exact split, then use the **same fit + eval path** in `tune_implement` and `tune_bake`.

## Pattern 1: tune scalar hyperparameters in place
```python
import pandas as pd
import skrub

data = pd.read_csv(skrub.datasets.fetch_toxicity().path).sample(frac=1.0, random_state=1)
X = skrub.X(data[["text"]])
y = skrub.y(data["is_toxic"])

encoder = skrub.MinHashEncoder(
    n_components=skrub.choose_int(5, 15, n_steps=5, name="N components")
)
classifier = YourClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)

pred = X.skb.apply(encoder).skb.apply(classifier, y=y)
print(pred.skb.describe_param_grid())
```

## Pattern 2: choose between encoder families
```python
n_components = skrub.choose_int(5, 15, name="N components")
encoder = skrub.choose_from(
    {
        "minhash": skrub.MinHashEncoder(n_components=n_components),
        "lse": skrub.StringEncoder(n_components=n_components),
    },
    name="encoder",
)
```

## Pattern 3: choose between model families
```python
from sklearn.linear_model import RidgeClassifier

model1 = YourModel1(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)
model2 = YourModel2(alpha=skrub.choose_float(0.01, 100, log=True, name="alpha"))
classifier = skrub.choose_from({"model1": model1, "model2": model2}, name="classifier")
pred = X.skb.apply(encoder).skb.apply(classifier, y=y)
```

## Search execution pattern
Holdout search (`tune_implement` only — holdout metric + `TUNING_BEST_PARAMS`):
```python
import json

search = pred.skb.make_randomized_search(
    n_iter=8, n_jobs=4, random_state=1, fitted=True
)
search.fit({"data": train_part})  # train fold only — not full train_df, not valid_part
best_learner = search.best_learner_

valid_pred = best_learner.predict({"data": valid_part})
rmse = mean_squared_error(valid_part[target_col].values, valid_pred) ** 0.5
print(f"Final Validation Performance: {rmse}")

print(pred.skb.describe_param_grid())  # inspect param names before bake, returns string
best_params = {}
for name, value in search.best_params_.items():
    if hasattr(value, "item"):
        value = value.item()
    best_params[name] = value
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
```
Tune agents should prefer explicit holdout `search.fit` above when structural code splits train/val.

## Anti-pattern vs correct pattern
- Anti-pattern (holdout leakage): `skrub.var("data", train_df)` + split + `make_learner(fitted=True)` + `predict({"data": valid_part})` for the metric line.
- Anti-pattern (fake tuning): define `choose_*` and then call only `pred.skb.make_learner(fitted=True)`.
- Anti-pattern (unresolved choose in estimator): `lr = skrub.choose_float(...)` then `Estimator(learning_rate=lr)` — many libraries copy/serialize kwargs at init and fail on skrub choice objects; keep `choose_*` on the DataOps apply path and resolve via search.
- Correct (inline on apply): `pred = X.skb.apply(Estimator(lr=skrub.choose_float(0.01, 0.1, name="lr")), y=y)` then `search = pred.skb.make_randomized_search(...)` — no intermediate variable holding a choice object.
- Anti-pattern (terminal tune crash): call `json.dumps(best_params)` on skrub/search params without `default=str` or numpy-to-Python conversion.
- Anti-pattern (incomparable scores): `search.fit({"data": train_part})` in tune_implement, then `make_learner(fitted=True)` on full `train_df` in tune_bake for the metric line.
- Anti-pattern (bake mapping): assume `search.best_params_` keys match `name=` strings — keys are often `data_op__0`, `data_op__1`, … Map **values** to estimator kwargs using plan `tunable_params` order (or `describe_param_grid()`), not key names.
- Correct holdout (early stages): bind `train_part`, print metric — no test/full-train block until submission.
- Correct tuning: define `choose_*`, run search (`make_randomized_search` / `make_grid_search`), then train/predict with best search result.
- Correct fixed-parameter run: no `choose_*`; use concrete parameter values directly.

## Refinement terminal tune (search → bake handoff)
- Runs once in the dedicated `tuning` pipeline stage after refinement completes.
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
- `describe_param_grid()` checked; bake literals mapped from `best_params_` values, not assumed key names.

## When to load other references
- Load `dataops_api_quickmap.md` for canonical DataOps pipeline shape and safe fit/predict patterns.
- Load `dataops_tuning_optuna.md` when using Optuna backend or trial-based search flows for tuning.
- Load `common_failure_fixes.md` when runtime errors appear, for fake-tuning, unresolved `choose_*` in estimator kwargs (#16), `choose_from` key-type, or scoring/debug issues.
- Load `encoding_skrub.md` when tuning scope includes encoding/preprocessing choices.
- Load `skrub_subsampling.md` when iteration speed is the bottleneck and subsampling is required.