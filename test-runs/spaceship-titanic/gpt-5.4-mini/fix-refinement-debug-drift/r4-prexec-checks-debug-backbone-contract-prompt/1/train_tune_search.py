
import json
import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# Load data
train_df = pd.read_csv("./input/train.csv")

target_col = "Transported"

# Honest holdout split
train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=42
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

# DataOps binding on train_part only
data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

# Minimal fix: vectorize categorical/string columns before CatBoost
# so CatBoost receives only numeric features.
vectorizer = skrub.TableVectorizer()

model_variant = skrub.choose_from(
    {
        "d5_lr0_03": CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            learning_rate=0.03,
            depth=5,
            random_seed=42,
            verbose=0,
        ),
        "d6_lr0_03": CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            learning_rate=0.03,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
        "d6_lr0_05": CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=0,
        ),
        "d7_lr0_05": CatBoostClassifier(
            loss_function="Logloss",
            iterations=300,
            learning_rate=0.05,
            depth=7,
            random_seed=42,
            verbose=0,
        ),
    },
    name="model_variant",
)

pred = X_train.skb.apply(vectorizer).skb.apply(model_variant, y=y_train)

search = pred.skb.make_randomized_search(
    n_iter=4, n_jobs=1, random_state=42, fitted=True
)
search.fit({"data": train_part})

valid_pred = search.best_learner_.predict({"data": valid_part})
valid_pred = np.asarray(valid_pred).ravel()
valid_pred_bool = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col].values, valid_pred_bool)
print(f"Final Validation Performance: {final_validation_score}")

chosen_variant = search.results_.iloc[0]["model_variant"]
variant_params = {
    "d5_lr0_03": {
        "iterations": 300,
        "learning_rate": 0.03,
        "depth": 5,
        "loss_function": "Logloss",
        "random_seed": 42,
        "verbose": 0,
    },
    "d6_lr0_03": {
        "iterations": 300,
        "learning_rate": 0.03,
        "depth": 6,
        "loss_function": "Logloss",
        "random_seed": 42,
        "verbose": 0,
    },
    "d6_lr0_05": {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 6,
        "loss_function": "Logloss",
        "random_seed": 42,
        "verbose": 0,
    },
    "d7_lr0_05": {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 7,
        "loss_function": "Logloss",
        "random_seed": 42,
        "verbose": 0,
    },
}
best_params = variant_params[str(chosen_variant)]
print("TUNING_BEST_PARAMS:", json.dumps(best_params, default=str))
