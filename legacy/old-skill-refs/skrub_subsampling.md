# Skrub Subsampling for faster development

This doc shows how to use `.skb.subsample()` to speed up interactive construction of a skrub DataOps plan by computing previews on a subsampled version of the original data.

By default subsampling is applied only for previews: the results shown when we display the plan, and the output of calling `.skb.preview()`. For other methods such as `.skb.get_learner()` or `.skb.cross_validate()`, no subsampling is done by default. We can explicitly ask for it with `keep_subsampling=True`. Even when `keep_subsampling=True`, subsampling is not applied to the `predict` method.

## Agent Verification Checklist (Skrub) - MODIFY THIS
Use this checklist when generating data preprocessing code:
- [ ] **No Pandas Engineering**: Did you use `skrub.TableVectorizer` to transform the dataframe to a vectorized representation?
- [ ] **DataOps Graph**: Did you define inputs using `skrub.X()`, `skrub.y()` or `skrub.var()` and `skrub.DataOp.skb.mark_as_X()`, `skrub.DataOp.skb.mark_as_y()`?
- [ ] **Relational Data**: If given multiple tables, did you use `skrub.Joiner` or `skrub.AggJoiner` instead of `pd.merge()`?

---

## 1. Lightweight construction of the DataOps plan on a subsample using skrub:
```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import skrub
import skrub.datasets

file_path = skrub.datasets.fetch_employee_salaries().path
dataset = pd.read_csv(file_path)
full_data = skrub.var("data", dataset)  # over 9k rows

# Subsample: Tell skrub to subsample the data when computing previews
data = full_data.skb.subsample(n=100)   # rest of plan will now use n=100 points for previews

# Define X and y
employees = data.drop(
    columns="current_annual_salary",
    errors="ignore",
).skb.mark_as_X()

salaries = data["current_annual_salary"].skb.mark_as_y()

# Apply TableVectorizer and then gradient boosting
predictions = employees.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(), y=salaries
)

# Turn on subsampling for other DataOps methods, such as `.skb.cross_validate()` -- helps quick error detection, but scores will be very low due to subsampling
predictions.skb.cross_validate(keep_subsampling=True)

# By default, when we do not explicitly ask for `keep_subsampling=True`, no subsampling takes place.
# Here we run cross-validation on the full data -- results in much better `test_score`
predictions.skb.cross_validate()
```


## When to load deeper references
- Multi-table joins/aggregations/entity relationships: load `multi_table_pipeline_pattern.md`.
- Choice/tuning logic and search-space composition: load `choices_hparam_pattern.md`.
- Runtime exceptions, shape/type mismatches, unresolved symbols: load `common_failure_fixes.md`.

## Web verification pattern (when uncertain)
Use targeted searches and patch only the uncertain line:
- `site:skrub-data.org <symbol_name>`
- `site:skrub-data.org DataOps <symbol_name>`
- `site:skrub-data.org reference data_ops`
