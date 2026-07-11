
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

random_state = 42
test_size = 0.2
target_cols = ["ConfirmedCases", "Fatalities"]

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

def add_date_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["dayofyear"] = df["Date"].dt.dayofyear
    df = df.drop(columns=["Date"])
    return df

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=test_size, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
data_train_fe = data_train.skb.apply_func(add_date_features)

X_train = data_train_fe.drop(columns=target_cols, errors="ignore").skb.mark_as_X()
y_train_cc = data_train_fe["ConfirmedCases"].skb.mark_as_y()

X_train_df = train_part.drop(columns=target_cols, errors="ignore").copy()
X_valid_df = valid_part.drop(columns=target_cols, errors="ignore").copy()

X_train_df = add_date_features(X_train_df)
X_valid_df = add_date_features(X_valid_df)

X_train_enc = pd.get_dummies(X_train_df, columns=["Province_State", "Country_Region"], dummy_na=True)
X_valid_enc = pd.get_dummies(X_valid_df, columns=["Province_State", "Country_Region"], dummy_na=True)
X_valid_enc = X_valid_enc.reindex(columns=X_train_enc.columns, fill_value=0)

X_train_enc = X_train_enc.astype(np.float32)
X_valid_enc = X_valid_enc.astype(np.float32)

model_cc = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
model_cc.fit(X_train_enc, train_part["ConfirmedCases"].values)

valid_pred_cc = model_cc.predict(X_valid_enc)
final_validation_score = mean_squared_error(
    valid_part["ConfirmedCases"].values, valid_pred_cc
) ** 0.5
print(f"Final Validation Performance: {final_validation_score}")
