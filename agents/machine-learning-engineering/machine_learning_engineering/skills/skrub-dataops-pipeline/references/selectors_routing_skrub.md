# Selectors and Column Routing with skrub

Use when preprocessing must differ by column subset, or when ablation/plan changes
which columns get which transformer. Keeps the DataOps graph intact.

## Core transformers
- `skrub.ApplyToCols(transformer, cols=selector, ...)`
- `skrub.SelectCols(cols=selector)`
- `skrub.DropCols(cols=selector)`
- `skrub.selectors as s` with `s.select(df, selector)` for inspection

## Selector quickmap
Combine with `&`, `|`, `-`, `^`, `~` (set logic).

| Selector | Use for |
|---|---|
| `s.all()` | all columns |
| `s.cols("a", "b")` | explicit names |
| `s.numeric()` / `s.string()` / `s.categorical()` | dtype families |
| `s.cardinality_below(n)` | low-cardinality categoricals |
| `s.has_nulls()` | columns with missing values |
| `s.glob("*_mm")` / `s.regex("pattern")` | name patterns |
| `s.filter(lambda col: ...)` | custom column rules |

Debug a rule before coding:
```python
selector = s.string() & s.cardinality_below(40)
selector.expand(train_df)  # column names that match
```

## sklearn inside skrub (allowed)
Standard sklearn transformers work inside `ApplyToCols` or `.skb.apply(...)`:
`StandardScaler`, `RobustScaler`, `SimpleImputer`, `OrdinalEncoder`, `PCA`, etc.

## DataOps routing pattern (split → transform → concat)
Use when different column groups need different preprocessing.
For holdout scripts, build this graph on `train_part` for the validation metric only.

```python
import skrub
from skrub import selectors as s

data_train = skrub.var("data", train_part)
X = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y = data_train[target_col].skb.mark_as_y()

high_card = s.string() - s.cardinality_below(40)
has_nulls = s.has_nulls()
leftover = s.all() - high_card - has_nulls

enc_high = X.skb.select(high_card).skb.apply(skrub.StringEncoder(n_components=8))
enc_nulls = X.skb.select(has_nulls).skb.apply(SimpleImputer(strategy="median"))
enc_rest = X.skb.select(leftover).skb.apply(skrub.TableVectorizer())

X_enc = enc_rest.skb.concat([enc_high, enc_nulls], axis=1)
model = X_enc.skb.apply(YourModel(), y=y)
```

Ordered categorical via plain map helper (keep DataOps-native):
```python
def encode_ordered(df):
    order = {"A": 3, "B": 2, "C": 1}
    return df["grade"].map(order)

enc_grades = data.skb.apply_func(encode_ordered)
```

## Drop uninformative columns with selectors
```python
from skrub import DropCols

DropCols(cols=~s.cardinality_below(3))  # drop high-cardinality/noisy cols
DropCols(cols=s.cols("households"))     # drop one redundant column
```

In DataOps:
```python
X_clean = X.skb.apply(DropCols(cols=s.cols("households")))
```

## ApplyToCols notes
- Single-column transformers (e.g. `StringEncoder`) run per matched column.
- Multi-column transformers (e.g. `PCA`) run on the selected subframe.
- `allow_reject=True` lets transformers like `ToDatetime()` skip non-applicable columns.

## Anti-patterns
- Hardcoding long column lists when a selector expresses the rule.
- Building `numeric_cols = train_part.select_dtypes(...)` before `.skb.apply_func(...)`, then
  `X.skb.select(numeric_cols)` — derived columns are dropped silently. Route with `s.numeric()`
  / `s.string()` on the post-FE `X` graph instead (see `feature_engineering_skrub.md`).
- Replacing the whole pipeline with sklearn `ColumnTransformer` outside DataOps.
- Applying one heavy encoder to all strings without cardinality routing.

## When to load other references
- Load `encoding_skrub.md` when choosing encoders after routing.
- Load `feature_engineering_skrub.md` for derived columns on routed paths.
- Load `dataops_api_quickmap.md` for learner/predict contracts.
