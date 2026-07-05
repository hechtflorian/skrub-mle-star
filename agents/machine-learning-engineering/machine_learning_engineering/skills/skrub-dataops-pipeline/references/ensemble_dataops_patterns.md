# Ensemble DataOps patterns

Use when **merging multiple model legs with varying FE or encoders**. Default: **Pattern A**.
Load for ensemble implement; also relevant when tuning one leg of a structural ensemble.

## When to use which

| Situation | Pattern |
|---|---|
| Merging existing solutions; varying FE/encoder per leg | **A** (default) |
| Custom blend, threshold, or calibration on predictions | **A** |
| Tuning one leg, freezing others | **A** |
| Fresh single-file script; simple soft vote; one submission learner | B (optional) |

## Rules (both patterns)
- Holdout only: `skrub.var("data", train_part)` → fit → `predict({"data": valid_part})`. No `test_df`, no full-train refit, no `submission.csv`.
- FE via plain `def` + `.skb.apply_func(...)`.
- Estimators via `.skb.apply(...)`; never `.skb.apply(plain_python_fn)`.
- **Never** invent custom classes/wrappers around graphs or fake `.skb` accessors.
- When merging input solutions: keep each leg's FE + encoder blocks intact; add only the blend layer.
- No `choose_*` in ensemble export — input solutions are already fixed-parameter.

---

## Pattern A (default): multi-leg + blend

One shared upstream DAG; each leg is its own `pred` chain; combine **predictions** (not graph nodes).

```
data_train → apply_func(fe) → X_train / y_train
                ├─ leg 1: encoder_a → model_a  → learner_a
                ├─ leg 2: encoder_b → model_b  → learner_b
                └─ leg N: encoder_n → model_n  → learner_n
                              ↓
                    numpy blend → metric
```

### Skeleton (ensemble agent)

Use structure as example guidance, not literals — reuse split, target, metric, FE helpers, encoders, and models from input solutions.

```python
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
# metric_fn, target_col, train_df — match input solutions

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()


def example_feature_eng(df):
    # Reuse FE helper(s) from input solutions (plain def, no @skrub.deferred)
    ...


def example_blended_pred(learner, df):
    try:
        pred = np.asarray(learner.predict_proba({"data": df}))
        return pred[:, 1] if pred.ndim == 2 and pred.shape[1] > 1 else pred.ravel()
    except Exception:
        return np.asarray(learner.predict({"data": df}), dtype=float).ravel()


# --- shared upstream (one DAG root) ---
data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(example_feature_eng)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

# --- leg 1: copy encoder + model from solution 1 ---
pred_1 = X_train.skb.apply(encoder_1).skb.apply(model_1, y=y_train)
learner_1 = pred_1.skb.make_learner(fitted=True)

# --- leg 2: copy encoder + model from solution 2 (may differ) ---
pred_2 = X_train.skb.apply(encoder_2).skb.apply(model_2, y=y_train)
learner_2 = pred_2.skb.make_learner(fitted=True)

# --- leg N ... ---

# --- blend (same rule as ensemble plan) ---
valid_p1 = example_blended_pred(learner_1, valid_part)
valid_p2 = example_blended_pred(learner_2, valid_part)
valid_blend = w1 * valid_p1 + w2 * valid_p2  # + ... ; threshold if plan requires
valid_pred = (valid_blend >= 0.5)  # or plan-specific rule

final_validation_score = metric_fn(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
```

---

## Pattern B (optional): VotingClassifier single learner

Use only for fresh scripts with simple soft voting. Wrap each leg in a sklearn `Pipeline`; one `.skb.apply(voting, y=y)`.

```python
from sklearn.base import clone
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

voting = VotingClassifier(
    estimators=[
        ("leg1", Pipeline([("enc", encoder_1), ("clf", model_1)])),
        ("leg2", Pipeline([("enc", clone(meta_pre)), ("clf", model_2)])),
    ],
    voting="soft",
    weights=[0.6, 0.4],
)
ensemble_pred = X_train.skb.apply(voting, y=y_train)
learner = ensemble_pred.skb.make_learner(fitted=True)
valid_pred = learner.predict({"data": valid_part})
```

Caveats: per-leg `Pipeline` required when encoders differ; `OneHotEncoder(sparse_output=False)` inside DataOps; no custom calibration/threshold; tuning one leg retrains the whole voter — **prefer Pattern A for tuning**.

## When to load other references
- `tuning_dataops_template.md` Pattern 3 — tune one leg, freeze others (Pattern A only).
- `common_failure_fixes.md` #21 — wrapper-class anti-pattern.
- `dataops_api_quickmap.md` — holdout bind rules.
- `submission_export.md` — full-train + test export (**submission agent only**).
