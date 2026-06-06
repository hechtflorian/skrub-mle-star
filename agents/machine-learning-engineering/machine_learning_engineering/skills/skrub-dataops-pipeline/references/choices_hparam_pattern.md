# Choice-Based Tuning Pattern (DataOps)

Use this reference when adding tunable parameters or model/encoder alternatives inside a DataOps graph.

## APIs quickmap
- `skrub.choose_int(low: int, high: int, ...) -> numeric choice: object`
- `skrub.choose_float(low: float, high: float, ...) -> numeric choice: object`
- `skrub.choose_from(outcomes: list or dict, name: str=None) -> choice: object`
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
```python
search = pred.skb.make_randomized_search(
    n_iter=8, n_jobs=4, random_state=1, fitted=True
)
best_learner = search.best_learner_
```

## Anti-pattern vs correct pattern
- Anti-pattern (fake tuning): define `choose_*` and then call only `pred.skb.make_learner(fitted=True)`.
- Correct tuning: define `choose_*`, run search (`make_randomized_search` / `make_grid_search`), then train/predict with best search result.
- Correct fixed-parameter run: no `choose_*`; use concrete parameter values directly.

## Checklist
- Tunables are in-graph, not external ad-hoc parameter dicts.
- Main path remains DataOps (`.skb.apply(...)` chain).
- Search object comes from the final prediction DataOp.
- For `choose_from({...})`, dictionary keys are readable outcome names and must be strings.
- If `choose_*` appears in final code, search execution is present and best search output is used.

## When to load other references
- Load `dataops_api_quickmap.md` for canonical DataOps pipeline shape and safe fit/predict patterns.
- Load `dataops_tuning_optuna.md` when using Optuna backend or trial-based search flows for tuning.
- Load `common_failure_fixes.md` when runtime errors appear, for fake-tuning, `choose_from` key-type, or scoring/debug issues.
- Load `encoding_skrub.md` when tuning scope includes encoding/preprocessing choices.
- Load `skrub_subsampling.md` when iteration speed is the bottleneck and subsampling is required.