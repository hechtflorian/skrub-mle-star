# Joining across multiple columns

Joining tables may be difficult if one entry on one side does not have an exact match on the other side.
`skrub.Joiner()` is a scikit-learn compatible transformer that enables performing joins across multiple keys, independently of the data type (numerical, string or mixed). The following example uses US domestic flights data to illustrate how space and time information from a pool of tables are combined for machine learning.



## Agent Verification Checklist (Skrub)
Use this checklist when generating join-heavy preprocessing code:
- [ ] **Join semantics**: Are `main_key` and `aux_key` aligned to the intended entity mapping?
- [ ] **Join tool choice**: Is `Joiner`/`AggJoiner` used when fuzzy or feature-augmentation joins are needed?
- [ ] **Leakage awareness**: Is target information excluded from auxiliary joins unless explicitly intended?
- [ ] **Pipeline continuity**: After join operations, does the flow continue cleanly into vectorization and learning?

## Quick Reference
- **Join transformer**: `Joiner(aux_table, main_key=[...], aux_key=[...])`
- **Feature augmentation**: `joined = joiner.fit_transform(main_df)`
- **Model pipeline**: `make_pipeline(joiner, TableVectorizer(), estimator)`
- **Common use case**: geospatial/time-based table augmentation before prediction

---

## 1. Lightweight construction of the DataOps plan on a subsample using skrub:
```python
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from skrub import Joiner
from skrub import TableVectorizer
from skrub.datasets import fetch_flight_delays

dataset = fetch_flight_delays()
flights = pd.read_csv(dataset.flights_path) # main table: flights dataset
# Sampling for faster computation.
flights = flights.sample(5_000, random_state=1, ignore_index=True)

airports = pd.read_csv(dataset.airports_path)   # airpot dataset from same database

weather = pd.read_csv(dataset.weather_path) # weather dataset from same database
# Sampling for faster computation.
weather = weather.sample(10_000, random_state=seed, ignore_index=True)

stations = pd.read_csv(dataset.stations_path)   # stations dataset

# Joining: feature augmentation across tables
aux = pd.merge(stations, weather, on="ID")

joiner = Joiner(airports, aux_key=["lat", "long"], main_key=["LATITUDE", "LONGITUDE"])

aux_augmented = joiner.fit_transform(aux)

joiner = Joiner(
    aux_augmented,
    aux_key=["YEAR/MONTH/DAY", "iata"],
    main_key=["Year_Month_DayofMonth", "Origin"],
)

flights.drop(columns=["TailNum", "FlightNum"])

# Training data passed through standard sklearn Pipeline
tv = TableVectorizer()
hgb = HistGradientBoostingClassifier()

pipeline_hgb = make_pipeline(joiner, tv, hgb)

y = flights["ArrDelay"]
X = flights.drop(columns=["ArrDelay"])
y = (y > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
pipeline_hgb.fit(X_train, y_train).score(X_test, y_test)
```
