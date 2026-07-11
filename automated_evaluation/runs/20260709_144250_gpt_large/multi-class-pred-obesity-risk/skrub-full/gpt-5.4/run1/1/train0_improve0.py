
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

random_state = 42
target_col = "NObeyesdad"

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

X_train_cat = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

X_train_lgbm = data_train[feature_cols].skb.mark_as_X()


vectorizer = skrub.TableVectorizer()
cat_model = CatBoostClassifier(
    random_state=random_state,
    verbose=0,
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

lgbm_model = Pipeline(
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

predictor_cat = X_train_cat.skb.apply(vectorizer).skb.apply(cat_model, y=y_train)
predictor_lgbm = X_train_lgbm.skb.apply(lgbm_model, y=y_train)

learner_cat = predictor_cat.skb.make_learner(fitted=True)
learner_lgbm = predictor_lgbm.skb.make_learner(fitted=True)

cat_proba = np.asarray(learner_cat.predict_proba({"data": valid_part}))
lgbm_proba = np.asarray(learner_lgbm.predict_proba({"data": valid_part}))

cat_weight = 0.9
lgbm_weight = 0.1
blended_proba = cat_weight * cat_proba + lgbm_weight * lgbm_proba

class_labels = learner_cat.classes_

cat_only_pred = class_labels[np.argmax(cat_proba, axis=1)]
valid_pred = class_labels[np.argmax(blended_proba, axis=1)]


final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
