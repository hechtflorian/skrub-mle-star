
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

DATA_DIR = "./input"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

for df in (train_df, test_df):
    df["Open Date"] = pd.to_datetime(df["Open Date"], format="%m/%d/%Y", errors="coerce")
    df["OpenYear"] = df["Open Date"].dt.year
    df["OpenMonth"] = df["Open Date"].dt.month
    df["OpenDay"] = df["Open Date"].dt.day
    reference_date = pd.Timestamp("2015-01-01")
    df["RestaurantAgeDays"] = (reference_date - df["Open Date"]).dt.days
    df.drop(columns=["Open Date"], inplace=True)

target_col = "revenue"
id_col = "Id"

X = train_df.drop(columns=[target_col])
y = np.log1p(train_df[target_col])
X_test = test_df.copy()

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=4,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

train_part, valid_part = train_test_split(train_df, test_size=0.2, random_state=42)

X_train_part = train_part.drop(columns=[target_col])
y_train_part = np.log1p(train_part[target_col])

X_valid_part = valid_part.drop(columns=[target_col])
y_valid_part = valid_part[target_col].values

pipeline.fit(X_train_part, y_train_part)

valid_pred_log = pipeline.predict(X_valid_part)
valid_pred = np.expm1(valid_pred_log)
valid_pred = np.maximum(valid_pred, 0)

final_validation_score = np.sqrt(mean_squared_error(y_valid_part, valid_pred))
print(f"Final Validation Performance: {final_validation_score}")

pipeline.fit(X, y)
test_pred_log = pipeline.predict(X_test)
test_pred = np.expm1(test_pred_log)
test_pred = np.maximum(test_pred, 0)

submission = pd.DataFrame({
    "Id": test_df[id_col],
    "Prediction": test_pred,
})

submission.to_csv("submission.csv", index=False)
print(submission.head())
