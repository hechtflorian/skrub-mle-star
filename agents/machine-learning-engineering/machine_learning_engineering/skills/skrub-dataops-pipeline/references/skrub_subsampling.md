# DataOps Subsampling for Fast Iteration

Use this pattern to speed up interactive development while keeping the final
evaluation on full data.

## Verified APIs (from local docs)
- `.skb.subsample(n=...)`
- `.skb.cross_validate(keep_subsampling=True)`
- `.skb.cross_validate()` (full data by default)

## Core pattern
```python
import pandas as pd
import skrub

dataset = pd.read_csv(skrub.datasets.fetch_employee_salaries().path)
full_data = skrub.var("data", dataset)

# Use only for previews / fast feedback loops.
data = full_data.skb.subsample(n=100)

X = data.drop(columns="current_annual_salary", errors="ignore").skb.mark_as_X()
y = data["current_annual_salary"].skb.mark_as_y()

pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(
    YourModel(), y=y
)

# Quick debug check on subsample:
quick_cv = pred.skb.cross_validate(keep_subsampling=True)

# Final evaluation runs on full data unless keep_subsampling=True:
full_cv = pred.skb.cross_validate()
```

## Rules
- Keep subsampling for fast iteration only.
- Do not report final quality metrics from subsampled CV.
- Run final search/evaluation on full data.

## When to load other references
- Load `dataops_api_quickmap.md` to keep subsampling within a correct DataOps flow.
- Load `encoding_skrub.md` if subsampling is used to evaluate encoding/preprocessing hypotheses quickly.
- Load `choices_hparam_pattern.md` when preview tuning is needed before larger search.
- Load `common_failure_fixes.md` if preview/final score behavior is inconsistent and runtime errors occur.