# Skrub Choices and Hyperparameter Pattern

This reference focuses on hyperparameter and component selection inside `skrub` DataOps plans using `choose_*` and `choose_from(...)`.
Use it when you need to tune encoder/model choices while keeping the pipeline declarative and DataOps-native.

`skrub.choose_from(...)` can tune hyperparameters, select optional configurations, and nest decisions.
Choices are not limited to model selection: they can be used anywhere DataOps are used (for example as arguments to DataOps methods and operators).

## Agent Verification Checklist (Skrub)
Use this checklist when generating tuning code:
- [ ] **Choice-based tuning**: Are tunables defined with `choose_int`, `choose_float`, `choose_bool`, or `choose_from`?
- [ ] **Graph integrity**: Are choices embedded inside DataOps operations (not external orchestration-only tuning)?
- [ ] **Search method alignment**: Is the DataOp search method (`make_grid_search`/`make_randomized_search`) used where appropriate?
- [ ] **Inference readiness**: Is the selected/best learner used for final predictions?

## Quick Reference
- **Scalar choices**: `skrub.choose_int(...)`, `skrub.choose_float(...)`, `skrub.choose_bool(...)`
- **Object choices**: `skrub.choose_from({...}, name="...")`
- **Apply with model**: `pred = X.skb.apply(encoder).skb.apply(model, y=y)`
- **Search from DataOp**: `pred.skb.make_grid_search(...)` / `pred.skb.make_randomized_search(...)`
- **Inspect search space**: `pred.skb.describe_param_grid()`

---

## 1. Basic example for hyperparameter tuning with skrub `choose_from` objects
```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import skrub
import skrub.datasets

file_path = skrub.datasets.fetch_toxicity().path    # load example dataset
data = pd.read_csv(file_path)

# This dataset is sorted -- all toxic tweets appear first, so we shuffle it
data = data.sample(frac=1.0, random_state=1)

texts = data[["text"]]
labels = data["is_toxic"]
X = skrub.X(texts)
y = skrub.y(labels)

# Use skrub.choose_from() within DataOps plan to let skrub create scikit-learn hyperparameter-tuner (e.g. GridSearchCV) automatically
encoder = skrub.MinHashEncoder(
    n_components=skrub.choose_int(5, 15, n_steps=5, name="N components")
)

classifier = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)
pred = X.skb.apply(encoder).skb.apply(classifier, y=y)

# Use `pred` DataOp to perform hyperparam search with `.skb.make_grid_search()` or `skb.make_randomized_search()` -- they accept same arguments as their scikit-learn counterparts (e.g. `scoring`, `cv`, `n_jobs`)
search = pred.skb.make_randomized_search(
    n_iter=8, n_jobs=4, random_state=1, fitted=True
)
# search.results_   # show results

# Retrieve best learner with best hyperparam config found during search -- use to make predictions on new data
best_learner = serach.best_learner_
```

## 2. Choosing between multiple possible encoders with skrub `choose_from` objects
```python
X, y = skrub.X(texts), skrub.y(labels)

n_components = skrub.choose_int(5, 15, name="N components")

encoder = skrub.choose_from(
    {
        "minhash": skrub.MinHashEncoder(n_components=n_components),
        "lse": skrub.StringEncoder(n_components=n_components),
    },
    name="encoder",
)
X.skb.apply(encoder, cols="text")
```

## 3. Choosing between multiple possible models with skrub `choose_from` objects
```python
from sklearn.linear_model import RidgeClassifier

hgb = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)
ridge = RidgeClassifier(alpha=skrub.choose_float(0.01, 100, log=True, name="α"))
classifier = skrub.choose_from({"hgb": hgb, "ridge": ridge}, name="classifier")
pred = X.skb.apply(encoder).skb.apply(classifier, y=y)
#print(pred.skb.describe_param_grid())

search = pred.skb.make_randomized_search(
    n_iter=16, n_jobs=4, random_state=1, fitted=True
)
#search.plot_results()
```
