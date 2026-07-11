
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

random_state = 42
target_col = "Personality"


def refine_survey_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    binary_cols = ["Stage_fear", "Drained_after_socializing"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def cast_category_to_object(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include=["category"]).columns
    for col in cat_cols:
        df[col] = df[col].astype("object")
    return df


def convert_object_to_codes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        if col != target_col:
            df[col] = pd.Categorical(df[col]).codes
            df[col] = df[col].replace(-1, np.nan)
    return df


train_path = "./input/train.csv"
train_df = pd.read_csv(train_path)

train_idx, valid_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.2,
    random_state=random_state,
    stratify=train_df[target_col],
)
train_part = train_df.iloc[train_idx].copy()
valid_part = train_df.iloc[valid_idx].copy()

data_train = skrub.var("data", train_part)
X_train = data_train.drop(columns=target_col, errors="ignore").skb.mark_as_X()
y_train = data_train[target_col].skb.mark_as_y()

vectorizer = skrub.TableVectorizer(low_cardinality=skrub.ToCategorical())

predictor = (
    X_train
    .skb.apply_func(refine_survey_table)
    .skb.apply(vectorizer)
    .skb.apply_func(cast_category_to_object)
    .skb.apply_func(convert_object_to_codes)
    .skb.apply(
        CatBoostClassifier(
            random_state=random_state,
            verbose=0,
        ),
        y=y_train,
    )
)

val_learner = predictor.skb.make_learner(fitted=True)
valid_pred = val_learner.predict({"data": valid_part})
final_validation_score = accuracy_score(valid_part[target_col], valid_pred)
print(f"Final Validation Performance: {final_validation_score}")
