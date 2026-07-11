
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor

random_state = 0
test_size = 0.2
target_col = "count"

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

def fe_func(df):
    out = df.copy()
    dt = pd.to_datetime(out["datetime"])
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    out = out.drop(columns=["datetime"], errors="ignore")
    return out

vectorizer = skrub.TableVectorizer()
model = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=random_state,
    verbose=-1,
)

pred = X_train.skb.apply_func(fe_func).skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.clip(np.asarray(valid_pred), 0, None)

final_validation_score = mean_squared_log_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
