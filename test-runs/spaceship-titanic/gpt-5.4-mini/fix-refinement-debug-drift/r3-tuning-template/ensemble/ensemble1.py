
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.isotonic import IsotonicRegression

INPUT_DIR = "./input"
train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

target_col = "Transported"

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

def prep(df):
    df = df.copy()
    cabin = df["Cabin"].astype(str).str.split("/", expand=True)
    df["CabinDeck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["CabinSide"] = cabin[2]
    df["NameLen"] = df["Name"].astype(str).str.len()
    df = df.drop(columns=["Cabin", "Name"], errors="ignore")
    df["CryoSleep"] = df["CryoSleep"].astype("float")
    df["VIP"] = df["VIP"].astype("float")
    df["HomePlanet"] = df["HomePlanet"].astype("category").cat.codes.replace(-1, np.nan)
    df["Destination"] = df["Destination"].astype("category").cat.codes.replace(-1, np.nan)
    df["CabinDeck"] = df["CabinDeck"].astype("category").cat.codes.replace(-1, np.nan)
    df["CabinSide"] = df["CabinSide"].astype("category").cat.codes.replace(-1, np.nan)
    return df

def fit_predict_split(train_df_local, valid_df_local, seed):
    data_train = skrub.var("data", train_df_local)
    X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
    y_train = data_train[target_col].skb.mark_as_y()

    model = RandomForestClassifier(random_state=42, n_estimators=300, n_jobs=-1)
    pred = X_train.skb.apply_func(prep).skb.apply(model, y=y_train)

    val_learner = pred.skb.make_learner(fitted=True)
    valid_prob = val_learner.predict({"data": valid_df_local})
    valid_prob = np.asarray(valid_prob).astype(float)

    test_prob = val_learner.predict({"data": test_df})
    test_prob = np.asarray(test_prob).astype(float)

    return valid_prob, test_prob

y_all = train_df[target_col].astype(int).values
oof_prob = np.zeros(len(train_df), dtype=float)
oof_count = np.zeros(len(train_df), dtype=int)

test_probs = []
seeds = [42, 52, 62, 72]
for seed in seeds:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr_idx, va_idx in skf.split(np.zeros(len(train_df)), y_all):
        tr_part = train_df.iloc[tr_idx].copy()
        va_part = train_df.iloc[va_idx].copy()
        va_prob, te_prob = fit_predict_split(tr_part, va_part, seed)
        oof_prob[va_idx] += va_prob
        oof_count[va_idx] += 1
        test_probs.append(te_prob)

oof_prob = oof_prob / np.maximum(oof_count, 1)

calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(oof_prob, y_all)

test_prob_avg = np.mean(np.vstack(test_probs), axis=0)
test_prob_cal = calibrator.transform(test_prob_avg)

oof_cal = calibrator.transform(oof_prob)
oof_pred = (oof_cal >= 0.5).astype(bool)
final_validation_score = accuracy_score(y_all, oof_pred)
print(f"Final Validation Performance: {final_validation_score}")

# Submission stage only: refit on full train and predict test with calibrated probabilities.
data_full = skrub.var("data", train_df)
X_full = data_full.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_full = data_full[target_col].skb.mark_as_y()

full_model = RandomForestClassifier(random_state=42, n_estimators=300, n_jobs=-1)
full_pred = X_full.skb.apply_func(prep).skb.apply(full_model, y=y_full)
full_learner = full_pred.skb.make_learner(fitted=True)

test_raw_prob = np.asarray(full_learner.predict({"data": test_df})).astype(float)
test_final_prob = calibrator.transform(test_raw_prob)
test_final_pred = (test_final_prob >= 0.5)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Transported": test_final_pred.astype(bool),
})
submission.to_csv("submission.csv", index=False)
