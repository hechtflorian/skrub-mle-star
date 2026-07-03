
import os
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

random_state = 42
target_col = "Transported"

train_path = os.path.join("./input", "train.csv")
test_path = os.path.join("./input", "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

for df in [train_df, test_df]:
    cabin_split = df["Cabin"].fillna("Z/0/Z").astype(str).str.split("/", expand=True)
    df["Deck"] = cabin_split[0]
    df["Num"] = cabin_split[1]
    df["Side"] = cabin_split[2]

    spend_cols = ["Spa", "VRDeck", "RoomService", "FoodCourt", "ShoppingMall"]
    df[spend_cols] = df[spend_cols].fillna(0)
    df["TotalSpend"] = df[spend_cols].sum(axis=1)
    df["NoSpend"] = (df["TotalSpend"] == 0).astype(int)
    df["IsMinor"] = (df["Age"].fillna(train_df["Age"].median()) < 18).astype(int)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=[target_col], errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.03,
    loss_function="Logloss",
    eval_metric="Accuracy",
    random_seed=random_state,
    verbose=0,
    allow_writing_files=False,
)

pred = X_train.skb.apply(vectorizer).skb.apply(model, y=y_train)
val_learner = pred.skb.make_learner(fitted=True)

valid_pred = val_learner.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
valid_labels = valid_pred > 0.5
final_validation_score = accuracy_score(valid_part[target_col].astype(int), valid_labels.astype(int))
print(f"Final Validation Performance: {final_validation_score}")
