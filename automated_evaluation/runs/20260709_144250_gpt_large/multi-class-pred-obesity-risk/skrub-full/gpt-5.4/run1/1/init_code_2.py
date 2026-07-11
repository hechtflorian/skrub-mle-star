
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import skrub
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

random_state = 42
target_col = "NObeyesdad"
id_col = "id"

train_path = os.path.join("./input", "train.csv")
train_df = pd.read_csv(train_path)

num_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
cat_cols = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS",
]
feature_cols = num_cols + cat_cols

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)

train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train[feature_cols].skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

model = Pipeline(
    steps=[
        ("prep", preprocess),
        (
            "clf",
            LGBMClassifier(
                objective="multiclass",
                num_class=7,
                n_estimators=700,
                learning_rate=0.05,
                num_leaves=31,
                random_state=random_state,
                verbose=-1,
            ),
        ),
    ]
)

predictor = X_train.skb.apply(model, y=y_train)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})

final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
