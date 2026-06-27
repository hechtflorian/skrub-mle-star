import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train_df = pd.read_csv("./input/train.csv")
test_df = pd.read_csv("./input/test.csv")
target_col = "median_house_value"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

@skrub.deferred
def add_ratio_features(df):
    out = df.copy()
    if "households" in out.columns:
        denom = out["households"].replace(0, np.nan)
        if "total_rooms" in out.columns:
            out["rooms_per_household"] = (
                out["total_rooms"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "total_bedrooms" in out.columns:
            out["bedrooms_per_household"] = (
                out["total_bedrooms"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if "population" in out.columns:
            out["population_per_household"] = (
                out["population"] / denom
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        out = out.drop(columns=["households"], errors="ignore")
    return out

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_ratio_features)
X_train = data_train_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train_fe[target_col].skb.mark_as_y()

model = CatBoostRegressor(
    loss_function="RMSE",
    iterations=500,
    learning_rate=0.05,
    depth=8,
    random_seed=42,
    verbose=0,
)

pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)

valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")

# Textbook graph
# Agent-discovered FE - @skrub.deferred ratio features from refinement (very likely influenced by skrub.TableReport)
# Honest holdout, no leakage
# short, ends at validation - no full-train refit
# recent version
# train1 = end of refinement loop, promoted script to tune & ensemble (optionally) => structural solution after refine step i => train1 promoted here, since better than init (and _improve)
# - train0 -> best init solution
# - refine: ablation.py -> plan + implement (inner loop) -> train0_improve0.py + train0_improve1.py
# - end of refinement outer loop -> train1.py (here; tablereport influence, before optional tune + ensemble)
# tuning will treat train1 as promoted baseline (if better than init and _improve)
# on this run, tune did not win, so train1 promoted to ensemble
# train1 here = promoted structural refine output (best of train0_improve*), not init script and not tuned/final submission.
# Good label = "refinement output - promoted structural dataops pipeline"
