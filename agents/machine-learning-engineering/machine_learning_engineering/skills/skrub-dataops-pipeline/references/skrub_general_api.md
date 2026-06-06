# Skrub General API (subset)

Use this reference for non-DataOps helper APIs around preprocessing and column logic.

## Core APIs quickmap
- `skrub.tabular_pipeline("regressor" or "classifier")`
- `skrub.TableVectorizer(cardinality_threshold: int=40, low_cardinality: transformer/"passthrough"/"drop", default=sklearn.OneHotEncoder, high_cardinality: transformer/"passthrough"/"drop", default=skrub.StringEncoder, numeric=PassThrough(), datetime=skrub.DatetimeEncoder(), specific_transformers=(), drop_null_fraction=1.0, drop_if_constant=False, drop_if_unique=False, datetime_format=None, null_strings=None, n_jobs=None)`
- `skrub.ApplyToCols(...)`
- `skrub.SelectCols(...)`
- `skrub.DropCols(...)`
- `skrub.TableReport(...)`
- `skrub.patch_display()`

## Example encoders/transformers
- `skrub.StringEncoder(...)`
- `skrub.MinHashEncoder(...)`
- `skrub.GapEncoder(...)`
- `skrub.ToCategorical(...)`

## Example selectors
- `skrub.selectors.regex(...)`
- `skrub.selectors.numeric()`
- `skrub.selectors.string()`
- `skrub.selectors.filter(...)`

## Quick patterns
### 1) Simple tabular baseline
```python
from skrub import tabular_pipeline

model = tabular_pipeline("regressor")
```

### 2) Build a custom sklearn pipeline around skrub.TableVectorizer
```python
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

pipeline = make_pipeline(TableVectorizer(), YourModel())
```

### 3) Apply transformers by column selection logic
```python
from sklearn.decomposition import PCA
from skrub import ApplyToCols, StringEncoder
from skrub import selectors as s

apply_string_encoder = ApplyToCols(
    StringEncoder(n_components=30),
    cols=["division", "employee_position_title"],
    rename_columns="lsa_{}",
)
apply_pca = ApplyToCols(PCA(n_components=8), cols=s.regex("lsa"))
```

### 4) Selector composition
```python
from skrub import selectors as s

low_cardinality = s.filter(lambda col: col.nunique() < 40)
high_cardinality = ~low_cardinality
selected = s.string() & low_cardinality
```

## When to load other references
- Load `dataops_api_quickmap.md` when the task requires DataOps graph creation (`var/X/y`, `.skb.apply`, learner execution).
- Load `encoding_skrub.md` for deeper encoder/preprocessing/routing strategy beyond API overview.
- Load `choices_hparam_pattern.md` and `dataops_tuning_optuna.md` when adding or debugging hyperparameter search.
- Load `joining_across_columns.md` for multi-table feature construction and merge logic.
- Load `common_failure_fixes.md` when runtime errors, API misuse, or parsing issues appear.
