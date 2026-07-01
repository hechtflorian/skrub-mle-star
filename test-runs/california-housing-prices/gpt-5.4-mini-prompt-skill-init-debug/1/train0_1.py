
import os
import warnings
import numpy as np
import pandas as pd
import skrub
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_data():
    train_path = os.path.join("./input", "train.csv")
    test_path = os.path.join("./input", "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def build_model():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=0)),
        ]
    )


def main():
    train_df, test_df = load_data()

    target_col = "median_house_value"
    X_df = train_df.drop(columns=[target_col])
    y_df = train_df[target_col]

    X = skrub.X(X_df)
    y = skrub.y(y_df)

    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
        X_df, y_df, test_size=0.2, random_state=42
    )

    model = build_model()
    model.fit(X_train_df, y_train_df)
    val_pred = model.predict(X_val_df)
    final_validation_score = rmse(y_val_df, val_pred)
    print(f"Final Validation Performance: {final_validation_score}")

    model.fit(X_df, y_df)
    test_pred = model.predict(test_df)

    submission = pd.DataFrame({"median_house_value": test_pred})
    submission.to_csv("submission.csv", index=False)
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
