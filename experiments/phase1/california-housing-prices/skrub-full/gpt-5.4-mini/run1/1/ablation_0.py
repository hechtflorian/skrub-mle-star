
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

def build_pipeline(df):
    data_train = skrub.var("data", df)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()
    pred = X_train.skb.apply(skrub.TableVectorizer()).skb.apply(
        HistGradientBoostingRegressor(random_state=42), y=y_train
    )
    return pred

baseline_pred = build_pipeline(train_part)
baseline_learner = baseline_pred.skb.make_learner(fitted=True)
baseline_valid_pred = baseline_learner.predict({"data": valid_part})
baseline_score = mean_squared_error(valid_part[target_col], baseline_valid_pred) ** 0.5
print(f"Ablation[baseline] RMSE: {baseline_score}")

ablated_pred = build_pipeline(train_part)
ablated_learner = ablated_pred.skb.make_learner(fitted=True)
ablated_valid_pred = ablated_learner.predict({"data": valid_part})
ablated_score = mean_squared_error(valid_part[target_col], ablated_valid_pred) ** 0.5
print(f"Ablation[ablated_variant] RMSE: {ablated_score}")

final_validation_score = ablated_score
print(f"Final Validation Performance: {final_validation_score}")
