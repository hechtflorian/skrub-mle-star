
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = "./input"
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

target_col = "median_house_value"
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

variants = {
    "baseline": False,
    "rooms_per_household": True,
}

scores = {}
best_variant = None
best_score = float("inf")

for variant_name, use_feature in variants.items():
    @skrub.deferred
    def add_features(df, use_feature=use_feature):
        out = df.copy()
        if use_feature:
            denom = out["households"].replace(0, np.nan) if "households" in out.columns else None
            if denom is not None:
                out["rooms_per_household"] = (
                    out["total_rooms"] / denom
                ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return out

    data_train = skrub.var("data", train_part)
    data_fe = data_train.skb.apply_func(add_features)
    X_train = data_fe.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_fe[target_col].skb.mark_as_y()

    model = HistGradientBoostingRegressor(random_state=42)
    pred = X_train.skb.apply(skrub.TableVectorizer(),).skb.apply(model, y=y_train)
    val_learner = pred.skb.make_learner(fitted=True)
    valid_pred = val_learner.predict({"data": valid_part})

    final_validation_score = mean_squared_error(valid_part[target_col], valid_pred) ** 0.5
    scores[variant_name] = final_validation_score
    print(f"Ablation[{variant_name}] RMSE: {final_validation_score}")

    if final_validation_score < best_score:
        best_score = final_validation_score
        best_variant = variant_name

print(f"Best ablation variant: {best_variant} | RMSE: {best_score}")
print(f"Final Validation Performance: {best_score}")
