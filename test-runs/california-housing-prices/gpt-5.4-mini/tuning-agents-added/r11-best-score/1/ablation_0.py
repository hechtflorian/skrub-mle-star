
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))

target_col = "median_house_value"
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def build_pred(data_df, ablate_variant=False):
    data = skrub.var("data", data_df)
    X = data.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y = data[target_col].skb.mark_as_y()

    if ablate_variant:
        model = HistGradientBoostingRegressor(random_state=42, max_depth=6, learning_rate=0.05)
    else:
        model = HistGradientBoostingRegressor(random_state=42, max_depth=8, learning_rate=0.07)

    pred = X.skb.apply(skrub.TableVectorizer()).skb.apply(model, y=y)
    return pred

baseline_pred = build_pred(train_part, ablate_variant=False)
baseline_learner = baseline_pred.skb.make_learner(fitted=True)
baseline_valid_pred = baseline_learner.predict({"data": valid_part})
baseline_score = mean_squared_error(valid_part[target_col], baseline_valid_pred) ** 0.5
print(f"Ablation[baseline] RMSE: {baseline_score}")

ablated_pred = build_pred(train_part, ablate_variant=True)
ablated_learner = ablated_pred.skb.make_learner(fitted=True)
ablated_valid_pred = ablated_learner.predict({"data": valid_part})
ablated_score = mean_squared_error(valid_part[target_col], ablated_valid_pred) ** 0.5
print(f"Ablation[ablated_variant] RMSE: {ablated_score}")

final_validation_score = min(baseline_score, ablated_score)
print(f"Final Validation Performance: {final_validation_score}")
