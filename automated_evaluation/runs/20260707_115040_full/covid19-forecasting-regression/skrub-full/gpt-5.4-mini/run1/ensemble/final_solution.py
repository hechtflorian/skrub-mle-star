
import os
import glob
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor

random_state = 42
target_cols = ["ConfirmedCases", "Fatalities"]

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Basic date features with minimal changes
for df in (train_df, test_df):
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["dayofweek"] = df["Date"].dt.dayofweek
    df["dayofyear"] = df["Date"].dt.dayofyear

# Fill missing categorical values consistently
for col in ["Province_State", "Country_Region"]:
    train_df[col] = train_df[col].fillna("")
    test_df[col] = test_df[col].fillna("")

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def rmsle(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.maximum(np.asarray(y_pred), 0)
    y_true = np.maximum(y_true, 0)
    return mean_squared_log_error(y_true, y_pred) ** 0.5

# DataOps graph built on train_part only for honest holdout validation
data_train = skrub.var("data", train_part)

# Fix: keep inference schema consistent by dropping Id from training features too
X_train = data_train.drop(columns=target_cols + ["Id"], errors="ignore").skb.mark_as_X()
y_cc_train = data_train["ConfirmedCases"].skb.mark_as_y()
y_ft_train = data_train["Fatalities"].skb.mark_as_y()

vectorizer_cc = skrub.TableVectorizer()
vectorizer_ft = skrub.TableVectorizer()

pred_cc = X_train.skb.apply(vectorizer_cc).skb.apply(
    RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=1,
    ),
    y=y_cc_train,
)

pred_ft = X_train.skb.apply(vectorizer_ft).skb.apply(
    RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=1,
    ),
    y=y_ft_train,
)

learner_cc = pred_cc.skb.make_learner(fitted=True)
learner_ft = pred_ft.skb.make_learner(fitted=True)

valid_features = valid_part.drop(columns=target_cols + ["Id"], errors="ignore")
valid_pred_cc = learner_cc.predict({"data": valid_features})
valid_pred_ft = learner_ft.predict({"data": valid_features})

score_cc = rmsle(valid_part["ConfirmedCases"], valid_pred_cc)
score_ft = rmsle(valid_part["Fatalities"], valid_pred_ft)
final_validation_score = (score_cc + score_ft) / 2.0
print(f"Final Validation Performance: {final_validation_score}")

# Fit final models on full training data for submission-style inference
data_full = skrub.var("data", train_df.copy())
X_full = data_full.drop(columns=target_cols + ["Id"], errors="ignore").skb.mark_as_X()
y_cc_full = data_full["ConfirmedCases"].skb.mark_as_y()
y_ft_full = data_full["Fatalities"].skb.mark_as_y()

full_pred_cc = X_full.skb.apply(vectorizer_cc).skb.apply(
    RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=1,
    ),
    y=y_cc_full,
)
full_pred_ft = X_full.skb.apply(vectorizer_ft).skb.apply(
    RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=1,
    ),
    y=y_ft_full,
)

full_learner_cc = full_pred_cc.skb.make_learner(fitted=True)
full_learner_ft = full_pred_ft.skb.make_learner(fitted=True)

# Smallest possible fix for the reported error:
# remove Id from test features so the schema matches training-time features.
test_features = test_df.drop(columns=["ForecastId", "Id"], errors="ignore")

test_pred_cc = np.maximum(full_learner_cc.predict({"data": test_features}), 0)
test_pred_ft = np.maximum(full_learner_ft.predict({"data": test_features}), 0)

submission = pd.DataFrame(
    {
        "ForecastId": test_df["ForecastId"],
        "ConfirmedCases": test_pred_cc,
        "Fatalities": test_pred_ft,
    }
)

os.makedirs("./final", exist_ok=True)
submission.to_csv("./final/submission.csv", index=False)
