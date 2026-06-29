# Encoding and Preprocessing with skrub

Use when the plan changes feature encoding or vectorization. For selector routing,
split/concat paths, or column drops, also load `selectors_routing_skrub.md`. For
ratios, redundancy, or cleaning, load `feature_engineering_skrub.md`.

## APIs quickmap
- Core baseline: `skrub.TableVectorizer(...)`, `skrub.tabular_pipeline(...)`
- String encoders: `StringEncoder` (default/fast), `GapEncoder` (interpretable),
  `MinHashEncoder` (fast/cheap), `TextEncoder` (slow, long text)
- Categorical: `ToCategorical`, sklearn `OrdinalEncoder` via `ApplyToCols`
- Datetime: `ToDatetime`, `DatetimeEncoder`
- Cleaning: `Cleaner`, `DropUninformative`
- Column routing: see `selectors_routing_skrub.md`

## TableVectorizer behavior
Dispatches by inferred column kind:
- numeric → cast to float
- datetime → `DatetimeEncoder`-style extraction
- low-cardinality categorical → one-hot style
- high-cardinality string/categorical → default `GapEncoder`

Built-in preprocessing: normalizes common missing strings; can parse date-like strings.

Diagnostics to guide encoding decisions with `vectorizer.*`: `kind_to_columns_`, `column_to_kind_`, `transformers_`,
`input_to_outputs_`, `all_processing_steps_`

## String encoder picker

| Encoder | Speed | Best for |
|---|---|---|
| `StringEncoder` | Fast | Default; short strings and moderate text |
| `GapEncoder` | Slow | Interpretable high-cardinality categories |
| `MinHashEncoder` | Very fast | Large data / quick ablations; lower accuracy |
| `TextEncoder` | Very slow | Long free text; needs extra deps |

Example tree-model speed preset:
```python
vectorizer = skrub.TableVectorizer(
    low_cardinality=skrub.ToCategorical(),
    high_cardinality=skrub.GapEncoder(),
)
```

## TableVectorizer kwargs (valid values)
**built-in defaults:** `low_cardinality`=OneHotEncoder instance; `high_cardinality`=StringEncoder instance.

For `low_cardinality` and `high_cardinality`, skrub accepts **only**:
- `"passthrough"` or `"drop"`
- or a **transformer instance** (e.g. `OneHotEncoder(...)`, `skrub.ToCategorical()`, `skrub.MinHashEncoder(...)`, `skrub.StringEncoder(...)`)

There is **no** string alias `"one-hot"`, `"auto"`, or `"default"` — those raise `ValueError: Value not understood`.

## Encoder tuning with `choose_*` (terminal tune / ablation follow-up)
Prefer a small `choose_from` grid of **whole vectorizers** or **encoder instances** — see `choices_hparam_pattern.md` Pattern 2b.

**A — route high-cardinality columns** (matches ablation `high_cardinality="drop"`):
```python
vectorizer = skrub.choose_from(
    {
        "default": skrub.TableVectorizer(),
        "drop_high": skrub.TableVectorizer(high_cardinality="drop"),
    },
    name="encoder_variant",
)
pred = X_train.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(model, y=y_train)
# then make_randomized_search on pred — see tuning_dataops_template.md
```

**B — tune the high-cardinality encoder** (skrub choices notebook pattern):
```python
n = skrub.choose_int(5, 15, name="n_components")
encoder = skrub.choose_from(
    {
        "minhash": skrub.MinHashEncoder(n_components=n),
        "string": skrub.StringEncoder(n_components=n),
    },
    name="encoder",
)
vectorizer = skrub.TableVectorizer(high_cardinality=encoder)
pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
```

**C — low-cardinality: pick transformer instances, not string labels:**
```python
from sklearn.preprocessing import OneHotEncoder

low_enc = skrub.choose_from(
    {"onehot": OneHotEncoder(handle_unknown="ignore"),
     "tocat": skrub.ToCategorical()},
    name="low_encoder",
)
vectorizer = skrub.TableVectorizer(low_cardinality=low_enc)
```

Map `TUNING_BEST_PARAMS` from `search.results_.iloc[0]["encoder_variant"]` (or the choice `name=`) plus variant literals — not raw `data_op__*` keys alone.

## Datetime handling
Parse strings first, then encode:

```python
from skrub import ToDatetime, DatetimeEncoder, ApplyToCols
import skrub.selectors as s

ApplyToCols(ToDatetime(), cols=s.any_date(), allow_reject=True)

# DataOps:
X_dt = X.skb.apply(ApplyToCols(ToDatetime(), allow_reject=True))
X_enc = X_dt.skb.apply(ApplyToCols(DatetimeEncoder(periodic_encoding="circular")))
```

`ToDatetime` learns format at fit time; failed parses become null. Use
`DatetimeEncoder` only after datetime dtypes exist.

## Selector-driven encoding (summary)
For split routing, deferred maps, and `.skb.concat`, load `selectors_routing_skrub.md`.
Quick example:
```python
low_card = s.string() & s.cardinality_below(40)
high_card = s.string() & ~s.cardinality_below(40)
X1 = X.skb.apply(ApplyToCols(OrdinalEncoder(), cols=low_card))
X2 = X1.skb.apply(ApplyToCols(skrub.StringEncoder(), cols=high_card))
```

## Decision policy
- Baseline `TableVectorizer()` for quick validation runs.
- If ablation flags encoding/preprocessing as impactful, change encoding block—not
  only model hyperparameters.
- Match encoder cost to column type (don't use `TextEncoder` on short categories).

## Anti-patterns
- Default `TableVectorizer()` while ablation shows encoding sensitivity.
- One heavy encoder on all string columns without cardinality routing.
- Applying `DatetimeEncoder` before parsing with `ToDatetime`.
- `TableVectorizer(low_cardinality="one-hot")` or `"auto"` — invalid; use a transformer instance or `"drop"`/`"passthrough"`.

## Checklist
- Encoding choice matches column types and model family.
- Datetime and missing-value handling verified (or delegated to `TableVectorizer`).
- Final path remains DataOps-first (`.skb.apply(...)` chain).

## When to load other references
- Load `selectors_routing_skrub.md` for multi-path routing, `DropCols`, split/concat.
- Load `feature_engineering_skrub.md` for ratios, redundancy drops, cleaning, scaling.
- Load `dataops_api_quickmap.md` for learner/predict contracts.
- Load `choices_hparam_pattern.md` when encoding options are tuned via `choose_*`.
- Load `common_failure_fixes.md` for pipeline breakages and holdout scoring issues.
