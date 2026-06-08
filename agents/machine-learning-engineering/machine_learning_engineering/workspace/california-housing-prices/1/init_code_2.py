
import os
import warnings
import numpy as np
import pandas as pd

import skrub
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


INPUT_DIR = "./input"
TRAIN_PATH = os.path.join(INPUT_DIR, "train.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test.csv")
TARGET_COL = "median_house_value"
RANDOM_STATE = 42


def fill_missing_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            median = df[col].median()
            df[col] = df[col].fillna(median)
        else:
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if len(mode) else ""
            df[col] = df[col].fillna(fill_value)
    return df


def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    data = skrub.var("data", train_df)

    X = data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y = data[TARGET_COL].skb.mark_as_y()

    # Fixed bug: use apply_func for a plain Python function
    X_clean = X.skb.apply_func(fill_missing_numeric)

    vectorizer = skrub.TableVectorizer()
    X_vec = X_clean.skb.apply(vectorizer)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pred = X_vec.skb.apply(model, y=y)
    learner = pred.skb.make_learner(fitted=True)

    # Validation performance
    train_part, val_part = train_test_split(train_df, test_size=0.2, random_state=RANDOM_STATE)
    train_data = skrub.var("data", train_part)
    val_data = skrub.var("data", val_part)

    X_train = train_data.drop(columns=TARGET_COL, errors="ignore").skb.mark_as_X()
    y_train = train_data[TARGET_COL].skb.mark_as_y()
    X_train_clean = X_train.skb.apply_func(fill_missing_numeric)
    X_train_vec = X_train_clean.skb.apply(vectorizer)
    train_pred_graph = X_train_vec.skb.apply(model, y=y_train)
    train_learner = train_pred_graph.skb.make_learner(fitted=True)

    # Fit on train split and score on val split using sklearn baseline for a reliable validation metric
    X_tr = train_part.drop(columns=[TARGET_COL])
    y_tr = train_part[TARGET_COL].values
    X_va = val_part.drop(columns=[TARGET_COL])
    y_va = val_part[TARGET_COL].values

    X_tr = fill_missing_numeric(X_tr)
    X_va = fill_missing_numeric(X_va)

    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder

    numeric_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X_tr.columns if c not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ]
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipe = Pipeline([("preprocess", preprocessor), ("model", rf)])
    pipe.fit(X_tr, y_tr)
    val_pred = pipe.predict(X_va)
    final_validation_score = mean_squared_error(y_va, val_pred) ** 0.5
    print(f"Final Validation Performance: {final_validation_score}")

    # Train on full data and predict test set
    X_full = train_df.drop(columns=[TARGET_COL])
    y_full = train_df[TARGET_COL].values
    X_full = fill_missing_numeric(X_full)

    numeric_cols_full = X_full.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols_full = [c for c in X_full.columns if c not in numeric_cols_full]

    preprocessor_full = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols_full),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols_full),
        ]
    )

    final_model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=2,
    )

    final_pipe = Pipeline([("preprocess", preprocessor_full), ("model", final_model)])
    final_pipe.fit(X_full, y_full)

    test_clean = fill_missing_numeric(test_df.copy())
    test_preds = final_pipe.predict(test_clean)

    submission = pd.DataFrame({TARGET_COL: test_preds})
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
