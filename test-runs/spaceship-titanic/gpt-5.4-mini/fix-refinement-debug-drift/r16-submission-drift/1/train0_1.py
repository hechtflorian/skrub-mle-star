
import os
import warnings
import numpy as np
import pandas as pd
import skrub

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

random_state = 42
target_col = "Transported"
train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.2, random_state=random_state
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer()

lgbm_model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("clf", LGBMClassifier(
            n_estimators=250,
            learning_rate=0.05,
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )),
    ]
)

pred_lgb = X_train.skb.apply(vectorizer).skb.apply(lgbm_model, y=y_train)
learner_lgb = pred_lgb.skb.make_learner(fitted=True)

valid_pred = learner_lgb.predict({"data": valid_part})
if hasattr(valid_pred, "to_numpy"):
    valid_pred = valid_pred.to_numpy()
valid_pred = np.asarray(valid_pred)
if valid_pred.dtype != bool:
    if valid_pred.ndim > 1:
        valid_pred = valid_pred[:, 1] if valid_pred.shape[1] > 1 else valid_pred.ravel()
    valid_pred = valid_pred > 0.5

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
