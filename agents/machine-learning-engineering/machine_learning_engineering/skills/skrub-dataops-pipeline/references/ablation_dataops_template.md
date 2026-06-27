# Ablation template (DataOps)

Minimal shape for refinement ablation scripts. **Copy structure, not literals** — reuse split, target, metric, and backbone from the input solution; use your own variant names and hypotheses.

## Rules
- Holdout only: same `train_test_split` as input; `skrub.var("data", train_part)` → fit → `predict({"data": valid_part})`. No `test_df`, no full-train refit.
- **Baseline variant** = unmodified pipeline (same backbone unless model swap is the hypothesis).
- **2–4 variants**, one change each; fixed params (no search unless you loaded `choices_hparam_pattern.md`).
- Every variant prints `Ablation[<name>] <metric>: <value>`; end with best-variant summary (see below).
- FE via plain `def fe_func(df): ...` + `.skb.apply_func(fe_func)` only — **not** `@skrub.deferred` on the same function, and not `apply_func(lambda df: ...)`. Estimators via `.skb.apply(...)`. Never `.skb.apply(plain_python_fn)`.
- For variant toggles (e.g. FE on/off), use separate `build_graph` functions — do not pass flags through a lambda into `apply_func`.
- Do not mutate DataOps nodes in place; do not redundantly pass an already declared DataOp into `skrub.var(...)`.
- Predict only through the fitted learner: `learner.predict({"data": valid_part})` — not raw columns into the estimator.

## Skeleton
```python
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
# metric_fn, target_col, train_df — match input solution

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=n, random_state=m
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def score_variant(variant_name, build_graph):
    """build_graph(data_train) -> pred chain (X/y marked, through encoder + estimator)."""
    data_train = skrub.var("data", train_part)
    pred = build_graph(data_train)
    learner = pred.skb.make_learner(fitted=True)
    valid_pred = learner.predict({"data": valid_part})
    score = metric_fn(valid_part[target_col], valid_pred)
    print(f"Ablation[{variant_name}] {metric_label}: {score}")
    return score


def baseline_graph(data_train):
    X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data_train[target_col].skb.mark_as_y()
    # optional: data_train = data_train.skb.apply_func(your_fe_func)
    encoder = skrub.TableVectorizer()  # or use your encoder from input solution
    return X.skb.apply(encoder).skb.apply(backbone_estimator, y=y)


scores = {}
scores["baseline"] = score_variant("baseline", baseline_graph)
# scores["no_fe"] = score_variant("no_fe", lambda d: baseline_graph(d))  # toggle flags inside
# scores["alt_encoder"] = score_variant("alt_encoder", alt_encoder_graph)
# scores["alt_backbone"] = score_variant("alt_backbone", alt_backbone_graph)  # only if model swap is the test

best_variant = min(scores, key=scores.get)  # max(...) if higher is better
best_score = scores[best_variant]
print(f"Best ablation variant: {best_variant} | {metric_label}: {best_score}")
```

Replace `backbone_estimator` with the **same estimator class and fixed params** as the input solution for baseline/non-model variants.

## Variant patterns (mix as needed)
| Hypothesis | Toggle |
|------------|--------|
| Feature block | `.skb.apply_func(fe_func)` on vs off |
| Column drop / routing | `DropCols`, selectors, or skip a FE step |
| Encoder | swap `TableVectorizer(...)` config or encoder block |
| Preprocessing step | omit or replace a `.skb.apply(...)` step |
| Backbone | different estimator or hyperparams — keep at least one baseline variant on original backbone |

Optional FE helper (plain function — `apply_func` already puts it in the graph):
```python
def fe_func(df):
    out = df.copy()
    # derived columns on out
    return out
```

## Print contract (required)
```
Ablation[baseline] <metric>: <value>
Ablation[<variant_n>] <metric>: <value>
Best ablation variant: <name> | <metric>: <value>
```
At least **three** `Ablation[...]` lines (baseline + two ablated). Use the task metric label.

## If a variant needs extra detail
Load only when that variant needs it and you need more context on how to use skrub for your ablation: `encoding_skrub.md`, `feature_engineering_skrub.md`, `selectors_routing_skrub.md`, `choices_hparam_pattern.md`.
