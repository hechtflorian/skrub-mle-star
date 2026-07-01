# Skrub DataOps: Building multi-table ML pipelines with skrub DataOps

High-level reference for building declarative tabular data preprocessing pipelines, relational joins, and DataOps graphs using the `skrub` library.

## Agent Verification Checklist (Skrub)
Use this checklist when generating data preprocessing code:
- [ ] **No Pandas Engineering**: Did you replace manual `pd.get_dummies()`, `.fillna()`, and `.apply()` with `skrub.TableVectorizer`?
- [ ] **DataOps Graph**: If building a tunable model, did you define inputs using `skrub.var()` or `skrub.X()`, `skrub.y()` and export via `.skb.make_learner()`?
- [ ] **Relational Data**: If given multiple tables, did you use `skrub.Joiner` or `skrub.AggJoiner` instead of `pd.merge()`?

## Quick Reference
- **Core Pipeline**: `skrub.tabular_pipeline(estimator)`
- **Auto-Encoder**: `TableVectorizer(high_cardinality="minhash")`
- **Fuzzy Joining**: `Joiner(aux_table, main_key="id", aux_key="id")`
- **DataOps Tuning**: `skrub.choose_from(["minhash", "one_hot"], name="encoder")`

---

## 1. Skrub API
Use these tools to build complex, declarative, and hyperparameter-tunable DAGs.

### 1.1 Building a pipeline:
* **`skrub.tabular_pipeline()`**: Get a simple machine-learning pipeline for tabular data.
* **`skrub.TableVectorizer()`**: Transform a dataframe to a numeric (vectorized) representation.
* **`skrub.ApplyToCols()`**: Apply a transformer to selected columns in a dataframe.
* **`skrub.SelectCols()`**: Select a subset of a DataFrame's columns.
* **`skrub.DropCols()`**: Drop a subset of a DataFrame's columns.

### 1.2 Encoding a column:
* **`skrub.StringEncoder()`**: Encode string features by using tf-idf vectorization and truncated singular value decomposition (SVD).
* **`skrub.TextEncoder()`**: Encode string features by applying a pretrained language model downloaded from the HuggingFace Hub.
* **`skrub.MinHashEncoder()`**: Encode string categorical features by applying the MinHash method to n-gram decompositions of strings.
* **`skrub.GapEncoder()`**: Encode string columns by constructing latent topics.
* **`skrub.SimilarityEncoder()`**: Encode string categories to a similarity matrix, to capture fuzziness across a few categories.
* **`skrub.ToCategorical()`**: Convert a string column to Categorical dtype.
* **`skrub.DatetimeEncoder()`**: Extract temporal features such as month, day of the week, … from a datetime column.
* **`skrub.ToDatetime()`**: Parse datetimes represented as strings and return `Datetime` columns.
* **`skrub.ToFloat()`**: Convert a column to 32-bit floating-point numbers.

### 1.3 Exploring a dataframe:
* **`skrub.TableReport()`**: Summarize the contents of a dataframe.
* **`skrub.column_associations()`**: Get measures of statistical associations between all pairs of columns.

### 1.4 Cleaning a dataframe:
* **`skrub.SquashingScaler()`**: Perform robust centering and scaling followed by soft clipping.
* **`skrub.deduplicate()`**: Deduplicate categorical data by hierarchically clustering similar strings.
* **`skrub.Cleaner()`**: Column-wise consistency checks and sanitization of dtypes, null values and dates.
* **`skrub.DropUninformative()`**: Drop column if it is found to be uninformative according to various criteria.

### 1.5 Joining dataframes:
* **`skrub.Joiner()`**: Augment features in a main table by fuzzy-joining an auxiliary table to it.
* **`skrub.AggJoiner()`**: Aggregate an auxiliary dataframe before joining it on a base dataframe.
* **`skrub.MultiAggJoiner()`**: Extension of the `AggJoiner` to multiple auxiliary tables.
* **`skrub.AggTarget()`**: Aggregate a target y before joining its aggregation on a base dataframe.
* **`skrub.InterpolationJoiner()`**: Join with a table augmented by machine-learning predictions.
* **`skrub.fuzzy_join()`**: Fuzzy (approximate) join.

### 1.6 Selectors:
* **`skrub.selectors.Selector()`**: Generic selector type, that returns set columns when applied.
* **`skrub.selectors.all()`**: Select all columns.
* **`skrub.selectors.any_date()`**: Select columns that have a Date or Datetime data type.
* **`skrub.selectors.boolean()`**: Select columns that have a Boolean data type.
* **`skrub.selectors.cardinality_below()`**: Select columns whose cardinality (number of unique values) is (strictly) below `threshold`.
* **`skrub.selectors.categorical()`**: Select columns that have a Categorical (or polars Enum) data type.
* **`skrub.selectors.cols()`**: Select columns by name.
* **`skrub.selectors.filter()`**: Select columns for which predicate returns True.
* **`skrub.selectors.filter_names()`**: Select columns based on their name.
* **`skrub.selectors.float()`**: Select columns that have a floating-point data type.
* **`skrub.selectors.glob()`**: Select columns by name with Unix shell style 'glob' pattern.
* **`skrub.selectors.has_dtype()`**: Select columns whose dtype is equal to one of the provided dtypes.
* **`skrub.selectors.has_nulls()`**: Select columns that contain at least one null value, or a proportion of null values above a given threshold.
* **`skrub.selectors.integer()`**: Select columns that have an integer data type.
* **`skrub.selectors.inv()`**: Invert a selector.
* **`skrub.selectors.make_selector()`**: Transform a selector, column name or list of column names into a selector.
* **`skrub.selectors.numeric()`**: Select columns that have a numeric data type.
* **`skrub.selectors.regex()`**: Select columns by name with a regular expression.
* **`skrub.selectors.select()`**: Apply a selector to a dataframe and return a dataframe with the selected columns.
* **`skrub.selectors.string()`**: Select columns that have a String data type.

### 1.7 DataOps (load `dataops_api_quickmap.md`)




## When to load deeper references
- Multi-table joins/aggregations/entity relationships: load `multi_table_pipeline_pattern.md`.
- Choice/tuning logic and search-space composition: load `choices_hparam_pattern.md`.
- Runtime exceptions, shape/type mismatches, unresolved symbols: load `common_failure_fixes.md`.

## Web verification pattern (when uncertain)
Use targeted searches and patch only the uncertain line:
- `site:skrub-data.org <symbol_name>`
- `site:skrub-data.org DataOps <symbol_name>`
- `site:skrub-data.org reference data_ops`
