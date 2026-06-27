# Feature Engineering and Preprocessing with skrub

Use when TableReport (or domain knowledge) suggests structural feature changes:
redundant columns, ratios, cleaning, scaling, or geo/datetime handling.
Planners, ablation, and implement agents may load this reference.
Keep the existing DataOps graph; change only the feature block under test.

## Planner priority (structural refinement)

- Prefer **one bounded structural change** per plan: derived features, encoding/routing,
  cleaning, or drop-one — before swapping model family or running search.
- Use the data profile selectively: correlated numerics, missingness, cardinality mix,
  coordinate-like column names.
- After any `apply_func` block, default to a **single** `TableVectorizer()` on post-FE `X`
  unless selectors clearly require split routing.

## Ablation contract (must follow)

- Ablate one structural block at a time (encoding, derived features, drop-one, scaling).
- Do not swap the backbone model (e.g. CatBoost → HGB) unless that is the explicit hypothesis.
- Pandas and sklearn are fine **inside** plain `apply_func` helpers or as transformers in
  `.skb.apply(...)`; the outer path stays DataOps (`var` / `mark_as_X` / `mark_as_y` / `.skb.apply`).
- **Holdout binding:** for each ablation variant, fit on `train_part` only (`skrub.var("data", train_part)`), then score on `valid_part`. Do not bind full `train_df` before the metric line (see `dataops_api_quickmap.md`).

## TableReport profile → ablation ideas

| Profile signal | Try (structural, fixed params) |
|---|---|
| `\|pearson\| >= 0.9` among count/size columns | drop one column; add ratio (e.g. col_a / col_b) |
| Strong geo pair (lat/lon) | keep both vs drop one; add interaction or distance block |
| `null_proportion > 0` | imputation path vs `DropUninformative`; selector on `s.has_nulls()` |
| `n_constant_columns > 0` | `Cleaner(drop_if_constant=True)` or `DropCols` |
| string/categorical columns | route encoders via selectors (see `selectors_routing_skrub.md`) |
| datetime-like strings | `ToDatetime` + `DatetimeEncoder` (see `encoding_skrub.md`) |

If redundancy ablations hurt validation, prefer keeping raw columns and tuning encoding/model instead.

## FE function pattern (one way only)

Use a **plain** Python function with `.skb.apply_func(...)` — do **not** also decorate it with `@skrub.deferred`. `apply_func` already keeps FE in the DataOps graph and passes a pandas `DataFrame` at run time.

```python
def add_features(df):
    out = df.copy()
    # column derivations on out
    return out

data_fe = data.skb.apply_func(add_features)
```

For ablation toggles (FE on/off, etc.), use **separate** graph builders or FE functions per variant — not `apply_func(lambda df: add_features(df, flag=...))`.

## Derived features (coordinate / geo) — DataOps-native

Prefer plain `def` + `.skb.apply_func` so features live in the graph.

Example for coordinate feature engineering:
```python
import numpy as np
import skrub

def add_coordinate_features(df):
    out = df.copy()
    cols = set(out.columns)
    lat_lon_pairs = (
        ({"latitude", "longitude"}, "latitude", "longitude"),
        ({"lat", "lon"}, "lat", "lon"),
        ({"x", "y"}, "x", "y"),
    )
    for required, lat_name, lon_name in lat_lon_pairs:
        if required.issubset(cols):
            lat = out[lat_name]
            lon = out[lon_name]
            out[f"{lat_name}_abs"] = lat.abs()
            out[f"{lon_name}_abs"] = lon.abs()
            out[f"{lat_name}_{lon_name}_interaction"] = lat * lon
            out["coord_radius"] = np.sqrt(lat ** 2 + lon ** 2)
            break
    return out

data = skrub.var("data", train_part)
data_fe = data.skb.apply_func(add_coordinate_features)
X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()
```

## Safe ratio / per-unit features (general)

Use when the profile shows highly correlated count or size columns. Always guard
division by zero and replace non-finite values.

```python
import numpy as np
import skrub

def add_ratio_features(df, numer_col, denom_col, out_name):
    out = df.copy()
    if numer_col not in out.columns or denom_col not in out.columns:
        return out
    denom = out[denom_col].replace(0, np.nan)
    ratio = (out[numer_col] / denom).replace([np.inf, -np.inf], np.nan)
    out[out_name] = ratio.fillna(0.0)
    return out

def add_common_ratios(df):
    out = df.copy()
    # Example: apply multiple ratios inside one deferred function when profile
    # suggests correlated count/size pairs (replace names with your columns).
    pairs = [
        ("col_a", "col_b", "col_a_per_col_b"),
        ("col_c", "col_d", "col_c_per_col_d"),
    ]
    for numer_col, denom_col, out_name in pairs:
        if numer_col in out.columns and denom_col in out.columns:
            denom = out[denom_col].replace(0, np.nan)
            out[out_name] = (
                (out[numer_col] / denom)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
    return out

data_fe = data.skb.apply_func(add_common_ratios)
```

Prefer one `apply_func` helper with several guarded ratios over many separate `apply_func`
calls when they share the same preprocessing block.

## Room / household ratios (example)

```python
import numpy as np
import skrub

def add_room_ratios(df):
    out = df.copy()
    denom = out["households"].replace(0, np.nan)
    out["rooms_per_household"] = (out["total_rooms"] / denom).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)
    return out

data_train = skrub.var("data", train_part)
data_fe = data_train.skb.apply_func(add_room_ratios)
X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()
# ... model + val_learner on train_part; predict valid_part for metric ...
```

Pandas `.assign(...)` inside the helper is equally valid for multi-column derivations.

## Post-FE vectorization (default after `apply_func`)

After any feature block, vectorize the full post-FE `X` graph unless selectors require split paths:

```python
data_fe = data.skb.apply_func(add_common_ratios)
X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_fe[target_col].skb.mark_as_y()
X_vec = X.skb.apply(skrub.TableVectorizer())
```

## Column lists after `apply_func` (common bug)

**Anti-pattern:** building column lists from raw pandas *before* FE, then selecting on `X`:

```python
# BAD: new columns from apply_func are invisible to numeric_cols
numeric_cols = train_df.select_dtypes(include=[np.number]).columns
data_fe = data.skb.apply_func(add_ratios)
X = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
X_num = X.skb.select(numeric_cols)  # misses derived columns
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
