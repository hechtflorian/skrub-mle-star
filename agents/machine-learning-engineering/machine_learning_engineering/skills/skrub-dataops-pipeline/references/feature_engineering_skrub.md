# Feature Engineering and Preprocessing with skrub

Use in ablation/refinement when TableReport (or domain knowledge) suggests structural
feature changes: redundant columns, ratios, cleaning, scaling, or geo/datetime handling.
Keep the existing DataOps graph; change only the feature block under test.

## Ablation contract (must follow)
- Ablate one structural block at a time (encoding, derived features, drop-one, scaling).
- Do not swap the backbone model (e.g. CatBoost → HGB) unless that is the explicit hypothesis.
- Pandas and sklearn are fine **inside** deferred functions or as transformers in
  `.skb.apply(...)`; the outer path stays DataOps (`var` / `mark_as_X` / `mark_as_y` / `.skb.apply`).

## TableReport profile → ablation ideas

| Profile signal | Try (structural, fixed params) |
|---|---|
| `\|pearson\| >= 0.9` among count/size columns | drop one column; add ratio (e.g. rooms/household) |
| Strong geo pair (lat/lon) | keep both vs drop one; add interaction or distance block |
| `null_proportion > 0` | imputation path vs `DropUninformative`; selector on `s.has_nulls()` |
| `n_constant_columns > 0` | `Cleaner(drop_if_constant=True)` or `DropCols` |
| string/categorical columns | route encoders via selectors (see `selectors_routing_skrub.md`) |
| datetime-like strings | `ToDatetime` + `DatetimeEncoder` (see `encoding_skrub.md`) |

If redundancy ablations hurt validation, prefer keeping raw columns and tuning encoding/model instead.

## Derived features (ratios, per-capita) — DataOps-native
Prefer `@skrub.deferred` + `.skb.apply_func` so features live in the graph.

Example:
```python
import numpy as np
import skrub

@skrub.deferred
def add_room_ratios(df):
    out = df.copy()
    denom = out["households"].replace(0, np.nan)
    out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return out

data = skrub.var("data", train_df)
data_fe = data.skb.apply_func(add_room_ratios)
X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()
```

Pandas `.assign(...)` inside a deferred function is equally valid for multi-column derivations.

## Column lists after `apply_func` (common bug)

**Anti-pattern:** building column lists from raw pandas *before* FE, then selecting on `X`:

```python
# BAD: new columns from apply_func are invisible to numeric_cols
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
data_fe = data.skb.apply_func(add_ratios)
X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
X_num = X.skb.select(numeric_cols)  # misses rooms_per_household, etc.
```

**Pattern 1 (preferred when unsure):** one encoder on the full FE graph:

```python
data_fe = data.skb.apply_func(add_ratios)
X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
X_vec = X.skb.apply(skrub.TableVectorizer())
```

**Pattern 2 (routing):** use skrub selectors on `X` *after* FE, not pre-FE pandas lists:

```python
import skrub.selectors as s

X_num = X.skb.select(s.numeric())
X_str = X.skb.select(s.string())
# or: X.skb.select(s.cols("rooms_per_household") | s.numeric())
```

Debug on a sample frame that includes derived columns if unsure: `s.select(sample_df, s.numeric())`.

## Redundancy and cleaning

### Drop one correlated column
```python
from skrub import DropCols

X_reduced = X.skb.apply(DropCols(cols=["households"]))
```

### Drop uninformative columns
`Cleaner` / `DropUninformative` remove constant or mostly-null columns (used by default inside
`TableVectorizer`; can also apply explicitly):

```python
from skrub import Cleaner, DropUninformative, ApplyToCols
import skrub.selectors as s

X_clean = X.skb.apply(Cleaner(drop_if_constant=True))
# or only on strings with high missingness:
X_clean = X.skb.apply(
    ApplyToCols(DropUninformative(drop_null_fraction=0.5), cols=s.string())
)
```

## Numeric scaling (outliers)
For numeric columns with heavy tails or infinities, use `SquashingScaler` via selectors:

```python
from skrub import ApplyToCols, SquashingScaler

X_scaled = X.skb.apply(
    ApplyToCols(SquashingScaler(max_absolute_value=3), cols=s.numeric())
)
```

Missing values are left as-is (not imputed by `SquashingScaler`).

## When to load other references
- Load `selectors_routing_skrub.md` for multi-path encoding/routing and `DropCols` selectors.
- Load `encoding_skrub.md` for string/datetime encoders and `TableVectorizer` tuning.
- Load `choices_hparam_pattern.md` only if ablation explicitly compares tuning vs fixed params.
- Load `common_failure_fixes.md` for target-drop and learner/predict contract errors.
